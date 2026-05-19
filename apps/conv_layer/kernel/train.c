//
// File:        train.c
// Description: Convolution-only training flow
// Author:      Haris Wang
// modified: Stavros Mitropoulos
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include "alexnet.h"
#include "runtime.h"
#include "weights.h"
#ifdef SPIKE
#include <printf.h>
#elif defined ARA_LINUX
#include <stdio.h>
#else
#include "printf.h"
#endif

#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))

// train.c is compiled with -O0 in this app profile; bind timer calls to
// runtime counter primitives to avoid unresolved inline symbols.
static inline int64_t alexnet_cycle_count_local(void)
{
    int64_t cycle_count = 0;
    asm volatile("fence; csrr %0, cycle" : "=r"(cycle_count));
    return cycle_count;
}

#define LEARNING_RATE 0.00001f

#ifndef ALEXNET_USE_MOMENTUM
#if defined(SPIKE)
#define ALEXNET_USE_MOMENTUM 0
#else
#define ALEXNET_USE_MOMENTUM 1
#endif
#endif

#ifndef ALEXNET_MAX_STEPS
#define ALEXNET_MAX_STEPS 0
#endif

#if defined(ALEXNET_LAYER_LOGS) && !defined(SPIKE)
#define ALEXNET_LOG_LAYER(...) printf_(__VA_ARGS__)
#else
#define ALEXNET_LOG_LAYER(...)
#endif

#define CONV_TOTAL_SAMPLES 4

static float d_conv1_weights[CONV1_WEIGHT_ELEMS];
static float d_conv1_bias[CONV1_OUT_CHANNELS];
static float v_conv1_weights[CONV1_WEIGHT_ELEMS];
static float v_conv1_bias[CONV1_OUT_CHANNELS];

static float d_conv1_output_buf[ALEXNET_STATIC_MAX_BATCH * CONV1_OUT_UNITS];
static float d_conv1_input_buf[ALEXNET_STATIC_MAX_BATCH * CONV1_IN_UNITS];

static float train_input_buf[ALEXNET_STATIC_MAX_BATCH * CONV1_IN_UNITS];
static float train_targets_buf[ALEXNET_STATIC_MAX_BATCH * CONV1_OUT_UNITS];

static int64_t last_loss_cycles = 0;
static int64_t last_zero_dinput_cycles = 0;
static int64_t last_backward_cycles = 0;
static int64_t last_update_cycles = 0;

static void zero_f32(float *buf, int n)
{
    memset(buf, 0, (size_t)n * sizeof(float));
}

static void zero_f32_vec(float *buf, int n)
{
    size_t max_vl;
    

    asm volatile("vsetvli %0, zero, e32, m8, ta, ma" : "=r"(max_vl));

    asm volatile("vmv.v.i v8, 0");

    float *ptr = buf;
    while (n > 0) {
        size_t vl;
        
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(n));

        asm volatile("vse32.v v8, (%0)" :: "r"(ptr));

        ptr += vl;
        n -= vl;
    }
}

static uint64_t checksum_bytes(const unsigned char *bytes, size_t len)
{
    uint64_t hash = 1469598103934665603ULL;
    for (size_t i = 0; i < len; i++) {
        hash ^= (uint64_t)bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

static uint64_t checksum_f32(const float *arr, size_t n)
{
    return checksum_bytes((const unsigned char *)arr, n * sizeof(float));
}

static uint64_t checksum_i32(const int *arr, size_t n)
{
    return checksum_bytes((const unsigned char *)arr, n * sizeof(int));
}


static float cross_entropy_loss(float *delta_preds, const float *preds, const int *labels, int units, int BATCH_SIZE)
{
    /**
     * Cross Entropy backward
     * 
     * Input:
     *      preds       [BATCH_SIZE, units]
     *      labels      [BATCH_SIZE]
     * Output:
     *      delta_preds [BATCH_SIZE, units]  (per-sample gradients)
     * */
    float ce_loss = 0;
    for (int p = 0; p < BATCH_SIZE; p++)
    {
        // find max for numerical stability (log-sum-exp trick)
        register float max_val = preds[p*units];
        for (int i = 1; i < units; i++)
            if (preds[i+p*units] > max_val) max_val = preds[i+p*units];

        register float esum = 0;
        for (int i = 0; i < units; i++)
            esum += exp(preds[i+p*units] - max_val);

        ce_loss += 0 - log(exp(preds[labels[p]+p*units] - max_val) / esum);

        for (int i = 0; i < units; i++)
        {
            if (labels[p] == i) {
                delta_preds[p * units + i] = exp(preds[i+p*units] - max_val) / esum - 1;
            }else {
                delta_preds[p * units + i] = exp(preds[i+p*units] - max_val) / esum;
            } 
        }
    }
    ce_loss /= BATCH_SIZE;
    ALEXNET_LOG_LAYER("cross entropy loss computed\n");
    return ce_loss;
}


static float v_fc1_weights[FC_INPUT_UNITS * FC_OUTPUT_UNITS];
static float v_fc1_bias[FC_OUTPUT_UNITS];



static inline void CLIP(float *x, float down, float up)
{
    *x = MIN(up, MAX(down, *x));
}

static void momentum_sgd(float *w, float *v_w, float *d_w, int units)
{
    for (int i = 0; i < units; i++) {
#if ALEXNET_USE_MOMENTUM
        v_w[i] = 0.9f * v_w[i] - LEARNING_RATE * d_w[i];
        CLIP(v_w + i, -1.0f, 1.0f);
        w[i] = w[i] + v_w[i];
#else
        (void)v_w;
        w[i] = w[i] - LEARNING_RATE * d_w[i];
#endif
    }
}

static void momentum_sgd_vec(float *w, float *v_w, const float *d_w, int units)
{
    float lr = LEARNING_RATE;
    int n = units;

#if ALEXNET_USE_MOMENTUM
    float momentum = 0.9f;
    float clip_min = -1.0f;
    float clip_max = 1.0f;

    while (n > 0) {
        size_t vl;
        
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(n));

        // v8: Velocity (v_w), v16: Gradients (d_w), v24: Weights (w)
        asm volatile("vle32.v v8,  (%0)" :: "r"(v_w));
        asm volatile("vle32.v v16, (%0)" :: "r"(d_w));
        asm volatile("vle32.v v24, (%0)" :: "r"(w));

        // 2. v_w = v_w * 0.9f
        asm volatile("vfmul.vf v8, v8, %0" :: "f"(momentum));

        // 3. v_w = v_w - LR * d_w
        // vfnmsac = Vector Floating-point Negative Multiply-Subtract Accumulate
        // vd = -(rs1 * vs2) + vd
        asm volatile("vfnmsac.vf v8, %0, v16" :: "f"(lr));

        // 4. CLIP(v_w, -1.0f, 1.0f)
        asm volatile("vfmax.vf v8, v8, %0" :: "f"(clip_min));
        asm volatile("vfmin.vf v8, v8, %0" :: "f"(clip_max));

        asm volatile("vfadd.vv v24, v24, v8");

        asm volatile("vse32.v v8,  (%0)" :: "r"(v_w));
        asm volatile("vse32.v v24, (%0)" :: "r"(w));

        w += vl;
        v_w += vl;
        d_w += vl;
        n -= vl;
    }
#else
    (void)v_w;
    while (n > 0) {
        size_t vl;
        asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(n));

        asm volatile("vle32.v v8,  (%0)" :: "r"(w));
        asm volatile("vle32.v v16, (%0)" :: "r"(d_w));

        // w = w - LR * d_w (Με χρήση της εντολής Negative MAC)
        asm volatile("vfnmsac.vf v8, %0, v16" :: "f"(lr));

        asm volatile("vse32.v v8,  (%0)" :: "r"(w));

        w += vl;
        d_w += vl;
        n -= vl;
    }
#endif
}


static void gradient_descent_a(void *argv)
{
    alexnet *net = (alexnet *)argv;
    if (net->trainable.fc1)
        momentum_sgd_vec(fc1_weights, v_fc1_weights, d_fc1_weights,
                     FC_INPUT_UNITS * FC_OUTPUT_UNITS);
}

static void gradient_descent_d(void *argv)
{
    alexnet *net = (alexnet *)argv;
    if (net->trainable.fc1)
        momentum_sgd_vec(fc1_bias, v_fc1_bias, d_fc1_bias,
                     FC_OUTPUT_UNITS);
}

static void gradient_descent(alexnet *net)
{
    if (!net->trainable.conv1) {
        return;
    }
    momentum_sgd(conv1_weights, v_conv1_weights, d_conv1_weights, CONV1_WEIGHT_ELEMS);
    momentum_sgd(conv1_bias, v_conv1_bias, d_conv1_bias, CONV1_OUT_CHANNELS);
}

void calloc_alexnet_d_params(alexnet *net)
{
    net->conv1.d_weights = d_conv1_weights;
    net->conv1.d_bias = d_conv1_bias;
    zero_f32(net->conv1.d_weights, CONV1_WEIGHT_ELEMS);
    zero_f32(net->conv1.d_bias, CONV1_OUT_CHANNELS);
}

void free_alexnet_d_params(alexnet *net)
{
    net->conv1.d_weights = NULL;
    net->conv1.d_bias = NULL;
}

static float mse_loss(float *delta_preds, const float *preds, const float *targets, int units, int BATCH_SIZE)
{
    /**
     * Mean Squared Error backward
     * * Input:
     * preds       [BATCH_SIZE, units]
     * targets     [BATCH_SIZE, units]
     * Output:
     * delta_preds [BATCH_SIZE, units] (per-sample gradients)
     * */
    float mse_loss_val = 0;

    for (int p = 0; p < BATCH_SIZE; p++)
    {
        for (int i = 0; i < units; i++)
        {
            int idx = p * units + i;
            
            float diff = preds[idx] - targets[idx];
            
            delta_preds[idx] = diff; 
            
            mse_loss_val += 0.5f * diff * diff;
        }
    }
    
    mse_loss_val /= (BATCH_SIZE * units);
    ALEXNET_LOG_LAYER("MSE loss computed: %f\n", mse_loss_val);
    return mse_loss_val;
}

static float mse_loss_vec(float *delta_preds, const float *preds, const float *targets, int units, int BATCH_SIZE)
{
    int total_elems = BATCH_SIZE * units;
    
    //$\sum (0.5 \cdot d^2) = 0.5 \cdot \sum (d^2)$

    float scale_factor = 0.5f / (float)total_elems;

    size_t max_vl; //vector length * 8 basically due LMUL=8
    asm volatile("vsetvli %0, zero, e32, m8, tu, ma" : "=r"(max_vl));

    asm volatile("vmv.v.i v8, 0");

    int n = total_elems;
    const float *p_ptr = preds;
    const float *t_ptr = targets;
    float *d_ptr = delta_preds;

    while (n > 0) {
        size_t vl;
        
        asm volatile("vsetvli %0, %1, e32, m8, tu, ma" : "=r"(vl) : "r"(n)); //we have set max_vl and compare with n=total_elements

        asm volatile("vle32.v v16, (%0)" :: "r"(p_ptr));
        asm volatile("vle32.v v24, (%0)" :: "r"(t_ptr));

        asm volatile("vfsub.vv v16, v16, v24");

        asm volatile("vse32.v v16, (%0)" :: "r"(d_ptr));

        asm volatile("vfmacc.vv v8, v16, v16");

        p_ptr += vl;
        t_ptr += vl;
        d_ptr += vl;
        n -= vl;
    }

    // reduction
    

    asm volatile("vsetvli zero, zero, e32, m8, tu, ma");   // set vl = VLMAX
    asm volatile("vmv.v.i v0, 0");                         // zero entire v0..v7 group
    asm volatile("vfredsum.vs v0, v8, v0");               // sum all elements of v8..v15 into v0[0]
    float sum_squares;
    asm volatile("vfmv.f.s %0, v0" : "=f"(sum_squares));

    float mse_loss_val = sum_squares * scale_factor;
    
    ALEXNET_LOG_LAYER("MSE loss computed: %f\n", mse_loss_val);
    return mse_loss_val;
}

void backward_alexnet(alexnet *net, const int *batch_Y, const float *batch_targets, float *loss_out)
{
    (void)batch_Y;

    if (net->batchsize > ALEXNET_STATIC_MAX_BATCH) {
        printf_("Error: batchsize %d exceeds static max batch %d\n", net->batchsize, ALEXNET_STATIC_MAX_BATCH);
        exit(1);
    }

    calloc_alexnet_d_params(net);

    float *curr_grad = d_conv1_output_buf;
    float *next_grad = d_conv1_input_buf;
    int total_out_elems = net->batchsize * net->conv1.out_units;
    int64_t t0 = 0;

    t0 = alexnet_cycle_count_local();
    float loss_val = mse_loss(curr_grad, net->conv1.output, batch_targets, total_out_elems);
    last_loss_cycles = alexnet_cycle_count_local() - t0;

    net->conv1.d_input = next_grad;
    t0 = alexnet_cycle_count_local();
    zero_f32(net->conv1.d_input, net->batchsize * net->conv1.in_units);
    last_zero_dinput_cycles = alexnet_cycle_count_local() - t0;
    net->conv1.d_output = curr_grad;

    t0 = alexnet_cycle_count_local();
    if (net->trainable.conv1) {
        conv_op_backward_full(&(net->conv1));
    } else {
        conv_op_backward_input_only(&(net->conv1));
    }
    last_backward_cycles = alexnet_cycle_count_local() - t0;

    t0 = alexnet_cycle_count_local();
    gradient_descent(net);
    last_update_cycles = alexnet_cycle_count_local() - t0;

    if (loss_out != NULL)
        *loss_out = loss_val;
}

void alexnet_train(alexnet *net, int epochs)
{
    if (net->batchsize > ALEXNET_STATIC_MAX_BATCH) {
        printf_("Error: batchsize %d exceeds static max batch %d\n", net->batchsize, ALEXNET_STATIC_MAX_BATCH);
        exit(1);
    }

    net->input = train_input_buf;
    float *batch_targets = train_targets_buf;

    int dataset_count = CONV_TOTAL_SAMPLES;
    int steps_per_epoch = dataset_count / net->batchsize;
    if (dataset_count % net->batchsize) steps_per_epoch++;
    if (steps_per_epoch <= 0) steps_per_epoch = 1;
    if (ALEXNET_MAX_STEPS > 0 && steps_per_epoch > ALEXNET_MAX_STEPS)
        steps_per_epoch = ALEXNET_MAX_STEPS;

    ALEXNET_LOG_LAYER("\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>> training begin >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n");
    for (int e = 0; e < epochs; e++) {
        printf_(">>>>>>>>>>>>>>>>>>>> epoch %d >>>>>>>>>>>>>>>>>>>>>>\n", e + 1);
        for (int b = 0; b < steps_per_epoch; b++) {
            float step_loss = 0.0f;
            int64_t prep_cycles = 0;
            int64_t forward_cycles = 0;
            int64_t backward_cycles = 0;
            int64_t t0 = 0;

            t0 = alexnet_cycle_count_local();
            int sample_offset = (b * net->batchsize) % dataset_count;
            memcpy(net->input,
                   test_inputs + sample_offset * CONV1_IN_UNITS,
                   (size_t)net->batchsize * CONV1_IN_UNITS * sizeof(float));
            memcpy(batch_targets,
                   test_targets + sample_offset * CONV1_OUT_UNITS,
                   (size_t)net->batchsize * CONV1_OUT_UNITS * sizeof(float));
            prep_cycles = alexnet_cycle_count_local() - t0;

            t0 = alexnet_cycle_count_local();
            forward_alexnet(net);
            forward_cycles = alexnet_cycle_count_local() - t0;

            t0 = alexnet_cycle_count_local();
            backward_alexnet(net, NULL, batch_targets, &step_loss);
            backward_cycles = alexnet_cycle_count_local() - t0;

            printf_("cycles[epoch %d batch %d/%d]: prep=%ld, forward=%ld, loss=%ld, zero_d_input=%ld, backward=%ld, update=%ld\n",
                    e + 1, b + 1, steps_per_epoch,
                    prep_cycles, forward_cycles,
                    last_loss_cycles, last_zero_dinput_cycles,
                    last_backward_cycles, last_update_cycles);
            printf_("epoch %d step %d/%d loss: %.6f\n", e + 1, b + 1, steps_per_epoch, step_loss);
        }
    }
    ALEXNET_LOG_LAYER(">>>>>>>>>>>>>>>>>>>>>>>>>>> training end >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n");
}

void alexnet_test(alexnet *net)
{
    if (net->batchsize > ALEXNET_STATIC_MAX_BATCH) {
        printf_("Error: batchsize %d exceeds static max batch %d\n", net->batchsize, ALEXNET_STATIC_MAX_BATCH);
        exit(1);
    }

    net->input = train_input_buf;
    memcpy(net->input,
           test_inputs,
           (size_t)net->batchsize * CONV1_IN_UNITS * sizeof(float));
    memcpy(train_targets_buf,
           test_targets,
           (size_t)net->batchsize * CONV1_OUT_UNITS * sizeof(float));

    forward_alexnet(net);
    float loss = mse_loss(d_conv1_output_buf, net->conv1.output, train_targets_buf,
                          net->batchsize * net->conv1.out_units);
    printf_("test loss: %.6f\n", loss);
}

void compute_batch_metrics(const int *preds, const int *labels, int batchsize)
{
    (void)preds;
    (void)labels;
    (void)batchsize;
}