//
// File:        train.c
// Description: Convolution-layer training flow
// Author:      Haris Wang
//
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

#ifndef ALEXNET_STATIC_MAX_BATCH
#ifdef ALEXNET_BATCHSIZE
#define ALEXNET_STATIC_MAX_BATCH ALEXNET_BATCHSIZE
#else
#define ALEXNET_STATIC_MAX_BATCH 4
#endif
#endif

#define CONV_TOTAL_SAMPLES 4
#define CONV1_PAD (CONV1_PADDING)
#define CONV1_PADDED_H (CONV1_IN_H + 2 * CONV1_PAD)
#define CONV1_PADDED_W (CONV1_IN_W + 2 * CONV1_PAD)
#define CONV1_PADDED_IN_UNITS (CONV1_IN_CHANNELS * CONV1_PADDED_H * CONV1_PADDED_W)
#define CONV1_XCOL_PER_IMG (CONV1_IN_CHANNELS * CONV1_KERNEL_L * CONV1_KERNEL_L * CONV1_OUT_W * CONV1_OUT_H)

static float d_conv1_weights[CONV1_WEIGHT_ELEMS];
static float d_conv1_bias[CONV1_OUT_CHANNELS];
static float v_conv1_weights[CONV1_WEIGHT_ELEMS];
static float v_conv1_bias[CONV1_OUT_CHANNELS];

static float d_conv1_output_buf[ALEXNET_STATIC_MAX_BATCH * CONV1_OUT_UNITS];
static float d_conv1_input_buf[ALEXNET_STATIC_MAX_BATCH * CONV1_IN_UNITS];
static float d_conv1_input_pad_buf[ALEXNET_STATIC_MAX_BATCH * CONV1_PADDED_IN_UNITS];

static float train_input_buf[ALEXNET_STATIC_MAX_BATCH * CONV1_IN_UNITS];
static float train_input_pad_buf[ALEXNET_STATIC_MAX_BATCH * CONV1_PADDED_IN_UNITS];
static float train_targets_buf[ALEXNET_STATIC_MAX_BATCH * CONV1_OUT_UNITS];

static float compare_output_ref[ALEXNET_STATIC_MAX_BATCH * CONV1_OUT_UNITS];
static float compare_output_pad[ALEXNET_STATIC_MAX_BATCH * CONV1_OUT_UNITS];
static float compare_input_col_buf[ALEXNET_STATIC_MAX_BATCH * CONV1_XCOL_PER_IMG];

static int64_t last_loss_cycles = 0;
static int64_t last_zero_dinput_cycles = 0;
static int64_t last_backward_cycles = 0;
static int64_t last_update_cycles = 0;
static conv_backward_cycle_breakdown last_conv_backward_breakdown = {0,0,0};
static int64_t last_conv_backward_total_cycles = 0;

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

static void pad_tensor(float *dst, const float *src, int batch, int channels, int in_h, int in_w, int pad)
{
    int out_h = in_h + 2 * pad;
    int out_w = in_w + 2 * pad;
    size_t total = (size_t)batch * (size_t)channels * (size_t)out_h * (size_t)out_w;
    memset(dst, 0, total * sizeof(float));

    int in_plane = in_h * in_w;
    int out_plane = out_h * out_w;
    for (int b = 0; b < batch; b++) {
        const float *src_b = src + b * channels * in_plane;
        float *dst_b = dst + b * channels * out_plane;
        for (int c = 0; c < channels; c++) {
            const float *src_c = src_b + c * in_plane;
            float *dst_c = dst_b + c * out_plane + pad * out_w + pad;
            for (int y = 0; y < in_h; y++) {
                memcpy(dst_c + y * out_w, src_c + y * in_w, (size_t)in_w * sizeof(float));
            }
        }
    }
}

static void unpad_tensor(float *dst, const float *src, int batch, int channels, int in_h, int in_w, int pad)
{
    int padded_h = in_h + 2 * pad;
    int padded_w = in_w + 2 * pad;
    int out_plane = in_h * in_w;
    int in_plane = padded_h * padded_w;
    for (int b = 0; b < batch; b++) {
        const float *src_b = src + b * channels * in_plane;
        float *dst_b = dst + b * channels * out_plane;
        for (int c = 0; c < channels; c++) {
            const float *src_c = src_b + c * in_plane + pad * padded_w + pad;
            float *dst_c = dst_b + c * out_plane;
            for (int y = 0; y < in_h; y++) {
                memcpy(dst_c + y * in_w, src_c + y * padded_w, (size_t)in_w * sizeof(float));
            }
        }
    }
}

static void compare_padding_paths(alexnet *net, const float *input_unpadded, const float *input_padded)
{
    (void)input_unpadded;

    conv_op old_op = net->conv1;
    old_op.input = (float *)input_padded;
    old_op.output = compare_output_ref;
    old_op.in_w = CONV1_PADDED_W;
    old_op.in_h = CONV1_PADDED_H;
    old_op.in_units = CONV1_PADDED_IN_UNITS;
    old_op.padding = 0;
    old_op.layer_id = 5;
    old_op.input_col = compare_input_col_buf;
    conv_op_forward_im2col(&old_op);

    conv_op new_op = old_op;
    new_op.output = compare_output_pad;
    new_op.input_col = NULL;
    conv_op_forward(&new_op);

    int total = net->batchsize * new_op.out_units;
    float max_diff = 0.0f;
    int max_idx = -1;
    for (int i = 0; i < total; i++) {
        float diff = fabsf(compare_output_ref[i] - compare_output_pad[i]);
        if (diff > max_diff) {
            max_diff = diff;
            max_idx = i;
        }
    }
    printf_("padding compare: max_abs_diff=%.6f idx=%d\n", max_diff, max_idx);
}

static float mse_loss_vec(float *delta_preds, const float *preds, const float *targets, int units, int batch_size)
{
    int total_elems = batch_size * units;
    float scale_factor = 0.5f / (float)total_elems;

    size_t max_vl;
    asm volatile("vsetvli %0, zero, e32, m8, tu, ma" : "=r"(max_vl));
    asm volatile("vmv.v.i v8, 0");

    int n = total_elems;
    const float *p_ptr = preds;
    const float *t_ptr = targets;
    float *d_ptr = delta_preds;

    while (n > 0) {
        size_t vl;
        asm volatile("vsetvli %0, %1, e32, m8, tu, ma" : "=r"(vl) : "r"(n));
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

    asm volatile("vsetvli zero, zero, e32, m8, tu, ma");
    asm volatile("vmv.v.i v0, 0");
    asm volatile("vfredsum.vs v0, v8, v0");
    float sum_squares;
    asm volatile("vfmv.f.s %0, v0" : "=f"(sum_squares));

    float mse_loss_val = sum_squares * scale_factor;
    ALEXNET_LOG_LAYER("MSE loss computed: %f\n", mse_loss_val);
    return mse_loss_val;
}

static inline void CLIP(float *x, float down, float up)
{
    *x = MIN(up, MAX(down, *x));
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

        asm volatile("vle32.v v8,  (%0)" :: "r"(v_w));
        asm volatile("vle32.v v16, (%0)" :: "r"(d_w));
        asm volatile("vle32.v v24, (%0)" :: "r"(w));

        asm volatile("vfmul.vf v8, v8, %0" :: "f"(momentum));
        asm volatile("vfnmsac.vf v8, %0, v16" :: "f"(lr));
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
        asm volatile("vfnmsac.vf v8, %0, v16" :: "f"(lr));
        asm volatile("vse32.v v8,  (%0)" :: "r"(w));

        w += vl;
        d_w += vl;
        n -= vl;
    }
#endif
}

static void gradient_descent(alexnet *net)
{
    if (!net->trainable.conv1)
        return;

    net->conv1.weights = conv1_weights;
    net->conv1.bias = conv1_bias;
    net->conv1.d_weights = d_conv1_weights;
    net->conv1.d_bias = d_conv1_bias;

    momentum_sgd_vec(conv1_weights, v_conv1_weights, d_conv1_weights, CONV1_WEIGHT_ELEMS);
    momentum_sgd_vec(conv1_bias, v_conv1_bias, d_conv1_bias, CONV1_OUT_CHANNELS);
}

void calloc_alexnet_d_params(alexnet *net)
{
    net->conv1.d_weights = d_conv1_weights;
    net->conv1.d_bias = d_conv1_bias;
    zero_f32_vec(net->conv1.d_weights, CONV1_WEIGHT_ELEMS);
    zero_f32_vec(net->conv1.d_bias, CONV1_OUT_CHANNELS);
}

void free_alexnet_d_params(alexnet *net)
{
    net->conv1.d_weights = NULL;
    net->conv1.d_bias = NULL;
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
    float *next_grad = d_conv1_input_pad_buf;
    float loss_val = 0.0f;
    int64_t t0 = 0;

    t0 = alexnet_cycle_count_local();
    loss_val = mse_loss_vec(curr_grad, net->conv1.output, batch_targets, net->conv1.out_units, net->batchsize);
    last_loss_cycles = alexnet_cycle_count_local() - t0;

    net->conv1.d_input = next_grad;
    t0 = alexnet_cycle_count_local();
    zero_f32_vec(net->conv1.d_input, net->batchsize * net->conv1.in_units);
    last_zero_dinput_cycles = alexnet_cycle_count_local() - t0;
    net->conv1.d_output = curr_grad;

    t0 = alexnet_cycle_count_local();
    if (net->trainable.conv1) {
        conv_op_backward_full_profile(&(net->conv1), &last_conv_backward_breakdown);
    } else {
        last_conv_backward_breakdown.d_input_cycles = 0;
        last_conv_backward_breakdown.d_bias_cycles = 0;
        last_conv_backward_breakdown.d_weights_im2col_cycles = 0;
        last_conv_backward_breakdown.d_weights_cycles = 0;
        conv_op_backward_input_only(&(net->conv1));
    }
    last_backward_cycles = alexnet_cycle_count_local() - t0;
    last_conv_backward_total_cycles = last_conv_backward_breakdown.d_input_cycles +
                                     last_conv_backward_breakdown.d_bias_cycles +
                                     last_conv_backward_breakdown.d_weights_cycles;

    unpad_tensor(d_conv1_input_buf, d_conv1_input_pad_buf,
                 net->batchsize, CONV1_IN_CHANNELS, CONV1_IN_H, CONV1_IN_W, CONV1_PAD);
    net->conv1.d_input = d_conv1_input_buf;

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

    net->input = train_input_pad_buf;
    float *batch_targets = train_targets_buf;
    net->conv1.padding = 0;
    net->conv1.in_w = CONV1_PADDED_W;
    net->conv1.in_h = CONV1_PADDED_H;
    net->conv1.in_units = CONV1_PADDED_IN_UNITS;

    int dataset_count = CONV_TOTAL_SAMPLES;
    int steps_per_epoch = dataset_count / net->batchsize;
    if (dataset_count % net->batchsize) steps_per_epoch++;
    if (steps_per_epoch <= 0) steps_per_epoch = 1;
    if (ALEXNET_MAX_STEPS > 0 && steps_per_epoch > ALEXNET_MAX_STEPS)
        steps_per_epoch = ALEXNET_MAX_STEPS;

    ALEXNET_LOG_LAYER("\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>> training begin >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n");
    for (int e = 0; e < epochs; e++) {
        printf_(">>>>>>>>>>>>>>>>>>>> epoch %d >>>>>>>>>>>>>>>>>>>>>>\n", e + 1);
        static int compare_once = 0;
        for (int b = 0; b < steps_per_epoch; b++) {
            float step_loss = 0.0f;
            int64_t prep_cycles = 0;
            int64_t forward_cycles = 0;
            int64_t t0 = 0;

            t0 = alexnet_cycle_count_local();
            int sample_offset = (b * net->batchsize) % dataset_count;
                 memcpy(train_input_buf,
                     test_inputs + sample_offset * CONV1_IN_UNITS,
                     (size_t)net->batchsize * CONV1_IN_UNITS * sizeof(float));
            memcpy(batch_targets,
                   test_targets + sample_offset * CONV1_OUT_UNITS,
                   (size_t)net->batchsize * CONV1_OUT_UNITS * sizeof(float));
                 pad_tensor(train_input_pad_buf, train_input_buf,
                      net->batchsize, CONV1_IN_CHANNELS, CONV1_IN_H, CONV1_IN_W, CONV1_PAD);
                 if (!compare_once) {
                  compare_padding_paths(net, train_input_buf, train_input_pad_buf);
                  compare_once = 1;
                 }
            prep_cycles = alexnet_cycle_count_local() - t0;

            t0 = alexnet_cycle_count_local();
            forward_alexnet(net);
            forward_cycles = alexnet_cycle_count_local() - t0;

            t0 = alexnet_cycle_count_local();
            backward_alexnet(net, NULL, batch_targets, &step_loss);
            (void)t0;

            printf_("cycles[epoch %d batch %d/%d]: prep=%ld, forward=%ld, loss=%ld, zero_d_input=%ld, backward=%ld, update=%ld\n",
                    e + 1, b + 1, steps_per_epoch,
                    prep_cycles, forward_cycles,
                    last_loss_cycles, last_zero_dinput_cycles,
                    last_backward_cycles, last_update_cycles);
                    printf_("conv backward breakdown: d_input=%ld, d_bias=%ld, d_weights_im2col=%ld, d_weights_total=%ld, total=%ld\n",
                    last_conv_backward_breakdown.d_input_cycles,
                    last_conv_backward_breakdown.d_bias_cycles,
                        last_conv_backward_breakdown.d_weights_im2col_cycles,
                    last_conv_backward_breakdown.d_weights_cycles,
                    last_conv_backward_total_cycles);
            printf_("epoch %d step %d/%d loss: %.6f\n", e + 1, b + 1, steps_per_epoch, step_loss);
        }
    }
    ALEXNET_LOG_LAYER(">>>>>>>>>>>>>>>>>>>>>>>>>>>> training end >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n");
}

void alexnet_test(alexnet *net)
{
    if (net->batchsize > ALEXNET_STATIC_MAX_BATCH) {
        printf_("Error: batchsize %d exceeds static max batch %d\n", net->batchsize, ALEXNET_STATIC_MAX_BATCH);
        exit(1);
    }

        net->input = train_input_pad_buf;
        net->conv1.padding = 0;
        net->conv1.in_w = CONV1_PADDED_W;
        net->conv1.in_h = CONV1_PADDED_H;
        net->conv1.in_units = CONV1_PADDED_IN_UNITS;
        memcpy(train_input_buf,
            test_inputs,
            (size_t)net->batchsize * CONV1_IN_UNITS * sizeof(float));
    memcpy(train_targets_buf,
           test_targets,
           (size_t)net->batchsize * CONV1_OUT_UNITS * sizeof(float));
        pad_tensor(train_input_pad_buf, train_input_buf,
             net->batchsize, CONV1_IN_CHANNELS, CONV1_IN_H, CONV1_IN_W, CONV1_PAD);

    forward_alexnet(net);
    float loss = mse_loss_vec(d_conv1_output_buf, net->conv1.output, train_targets_buf,
                              net->conv1.out_units, net->batchsize);
    printf_("test loss: %.6f\n", loss);
}

void compute_batch_metrics(const int *preds, const int *labels, int batchsize)
{
    (void)preds;
    (void)labels;
    (void)batchsize;
}