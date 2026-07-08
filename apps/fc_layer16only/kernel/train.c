//
// File:        train.c
// Description: Pure FP16 training for fc_layer16only
// Author:      Haris Wang
// FP16 refactor: Stavros Mitropoulos, NTUA
//
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
// #include <assert.h>
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

#ifdef SHOW_OP_TIME
#ifndef ALEXNET_TIMER_HZ
#define ALEXNET_TIMER_HZ 1000000000ULL
#endif

#ifndef ALEXNET_USE_RDCYCLE_TIMER
#define ALEXNET_USE_RDCYCLE_TIMER 0
#endif

typedef struct {
    uint64_t tv_sec;
    uint64_t tv_nsec;
} alexnet_timer_t;

static inline void alexnet_timer_now(alexnet_timer_t *tp)
{
#if ALEXNET_USE_RDCYCLE_TIMER
    uint64_t cycles = 0;
    asm volatile ("rdcycle %0" : "=r"(cycles));
    tp->tv_sec = cycles / ALEXNET_TIMER_HZ;
    tp->tv_nsec = ((cycles % ALEXNET_TIMER_HZ) * 1000000000ULL) / ALEXNET_TIMER_HZ;
#else
    static uint64_t soft_ticks = 0;
    soft_ticks++;
    tp->tv_sec = soft_ticks / ALEXNET_TIMER_HZ;
    tp->tv_nsec = soft_ticks % ALEXNET_TIMER_HZ;
#endif
}
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
#define ALEXNET_STATIC_MAX_BATCH 2
#endif
#endif

#define FC_INPUT_UNITS  2048
#define FC_OUTPUT_UNITS 512
#ifndef FC_TOTAL_SAMPLES
#define FC_TOTAL_SAMPLES 4
#endif

#if defined(SET_CEL) && defined(SET_MSE)
#error "Define only one loss flag: SET_CEL or SET_MSE"
#endif

#if !defined(SET_CEL) && !defined(SET_MSE)
#define SET_MSE
#endif

static _Float16 d_fc1_weights[FC_INPUT_UNITS * FC_OUTPUT_UNITS];
static _Float16 d_fc1_bias[FC_OUTPUT_UNITS];
static _Float16 d_fc1_output_buf[ALEXNET_STATIC_MAX_BATCH * FC_OUTPUT_UNITS];
static _Float16 mse_targets_buf[ALEXNET_STATIC_MAX_BATCH * FC_OUTPUT_UNITS];

#define ALEXNET_MAX_BACKWARD_UNITS \
    MAX(MAX(MAX(MAX(MAX(MAX(MAX(MAX(MAX(MAX(MAX(MAX( \
        FC7_LAYER, \
        FC6_LAYER), \
        C5_CHANNELS * POOLING5_L * POOLING5_L), \
        C5_CHANNELS * FEATURE5_L * FEATURE5_L), \
        C4_CHANNELS * FEATURE5_L * FEATURE5_L), \
        C4_CHANNELS * FEATURE4_L * FEATURE4_L), \
        C3_CHANNELS * FEATURE4_L * FEATURE4_L), \
        C3_CHANNELS * FEATURE3_L * FEATURE3_L), \
        C2_CHANNELS * POOLING2_L * POOLING2_L), \
        C2_CHANNELS * FEATURE2_L * FEATURE2_L), \
        C1_CHANNELS * POOLING1_L * POOLING1_L), \
        C1_CHANNELS * FEATURE1_L * FEATURE1_L), \
        IN_CHANNELS * FEATURE0_L * FEATURE0_L)

static _Float16 d_grad_ping_0[ALEXNET_STATIC_MAX_BATCH * FC_INPUT_UNITS];

static _Float16 train_input_buf[ALEXNET_STATIC_MAX_BATCH * FC_INPUT_UNITS];
static int      train_batch_Y_buf[ALEXNET_STATIC_MAX_BATCH];
static int      train_preds_buf[ALEXNET_STATIC_MAX_BATCH];

static _Float16 test_input_buf[ALEXNET_STATIC_MAX_BATCH * FC_INPUT_UNITS];
static int      test_batch_Y_buf[ALEXNET_STATIC_MAX_BATCH];
static int      test_preds_buf[ALEXNET_STATIC_MAX_BATCH];

static int metrics_true_pos[FC_OUTPUT_UNITS];
static int metrics_false_pos[FC_OUTPUT_UNITS];
static int metrics_false_neg[FC_OUTPUT_UNITS];

static int64_t last_loss_cycles              = 0;
static int64_t last_zero_dinput_cycles       = 0;
static int64_t last_fc_backward_total_cycles = 0;
static fc_backward_cycle_breakdown last_fc_backward_breakdown = {0, 0, 0};

static void zero_f16(_Float16 *buf, int n)
{
    memset(buf, 0, (size_t)n * sizeof(_Float16));
}

static void zero_f16_vec(_Float16 *buf, int n)
{
    _Float16 *ptr = buf;
    while (n > 0) {
        size_t vl;
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(n));
        asm volatile("vmv.v.i v8, 0");
        asm volatile("vse16.v v8, (%0)" :: "r"(ptr));
        ptr += vl;
        n -= vl;
    }
}

static void vcopy_f16(_Float16 *dst, const _Float16 *src, size_t n)
{
    while (n > 0) {
        size_t vl;
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(n));
        asm volatile("vle16.v v8, (%0)" :: "r"(src) : "memory");
        asm volatile("vse16.v v8, (%0)" :: "r"(dst) : "memory");
        src += vl; dst += vl; n -= vl;
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

static uint64_t checksum_f16(const _Float16 *arr, size_t n)
{
    return checksum_bytes((const unsigned char *)arr, n * sizeof(_Float16));
}

static uint64_t checksum_i32(const int *arr, size_t n)
{
    return checksum_bytes((const unsigned char *)arr, n * sizeof(int));
}


static float cross_entropy_loss_16(_Float16 *delta_preds, const _Float16 *preds,
                                    const int *labels, int units, int BATCH_SIZE)
{
    float ce_loss = 0;
    for (int p = 0; p < BATCH_SIZE; p++)
    {
        register float max_val = (float)preds[p * units];
        for (int i = 1; i < units; i++)
            if ((float)preds[i + p*units] > max_val)
                max_val = (float)preds[i + p*units];

        register float esum = 0;
        for (int i = 0; i < units; i++)
            esum += expf((float)preds[i + p*units] - max_val);

        ce_loss += -logf(expf((float)preds[labels[p] + p*units] - max_val) / esum);

        for (int i = 0; i < units; i++)
        {
            float softmax_i = expf((float)preds[i + p*units] - max_val) / esum;
            delta_preds[p * units + i] = (_Float16)(softmax_i - (labels[p] == i ? 1.0f : 0.0f));
        }
    }
    ce_loss /= BATCH_SIZE;
    ALEXNET_LOG_LAYER("cross entropy loss computed\n");
    return ce_loss;
}


static float mse_loss_16(_Float16 *delta_preds, const _Float16 *preds,
                          const _Float16 *targets, int units, int BATCH_SIZE)
{
    float mse_loss_val = 0;

    for (int p = 0; p < BATCH_SIZE; p++)
    {
        for (int i = 0; i < units; i++)
        {
            int   idx  = p * units + i;
            float diff = (float)preds[idx] - (float)targets[idx];
            delta_preds[idx] = (_Float16)diff;
            mse_loss_val += 0.5f * diff * diff;
        }
    }

    mse_loss_val /= (BATCH_SIZE * units);
    ALEXNET_LOG_LAYER("MSE loss computed: %f\n", mse_loss_val);
    return mse_loss_val;
}

static float mse_loss_vec_16(_Float16 *delta_preds, const _Float16 *preds,
                              const _Float16 *targets, int units, int BATCH_SIZE)
{
    int   total_elems  = BATCH_SIZE * units;
    float scale_factor = 0.5f / (float)total_elems;

    size_t max_vl;
    asm volatile("vsetvli %0, zero, e16, m8, tu, ma" : "=r"(max_vl));
    asm volatile("vmv.v.i v8, 0");

    int n = total_elems;
    const _Float16 *p_ptr = preds;
    const _Float16 *t_ptr = targets;
    _Float16       *d_ptr = delta_preds;

    while (n > 0) {
        size_t vl;
        asm volatile("vsetvli %0, %1, e16, m8, tu, ma" : "=r"(vl) : "r"(n));

        asm volatile("vle16.v v16, (%0)" :: "r"(p_ptr));
        asm volatile("vle16.v v24, (%0)" :: "r"(t_ptr));

        asm volatile("vfsub.vv v16, v16, v24");
        asm volatile("vse16.v v16, (%0)" :: "r"(d_ptr));
        asm volatile("vfmacc.vv v8, v16, v16");

        p_ptr += vl;
        t_ptr += vl;
        d_ptr += vl;
        n     -= vl;
    }

    asm volatile("vsetvli zero, zero, e16, m8, tu, ma");
    asm volatile("vmv.v.i v0, 0");
    asm volatile("vfredsum.vs v0, v8, v0");

    /* Extract v0[0] via memory to avoid FP16 f-register NaN-boxing issues */
    _Float16 sum_scratch;
    size_t one = 1;
    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" :: "r"(one));
    asm volatile("vse16.v v0, (%0)" :: "r"(&sum_scratch) : "memory");

    float mse_loss_val = (float)sum_scratch * scale_factor;

    ALEXNET_LOG_LAYER("MSE loss computed: %f\n", mse_loss_val);
    return mse_loss_val;
}


void calloc_alexnet_d_params_16(alexnet *net)
{
    net->fc1.d_weights = d_fc1_weights;
    net->fc1.d_bias    = d_fc1_bias;
    zero_f16_vec(net->fc1.d_weights, net->fc1.in_units * net->fc1.out_units);
    zero_f16_vec(net->fc1.d_bias,    net->fc1.out_units);
}

void free_alexnet_d_params_16(alexnet *net)
{
    net->fc1.d_weights = NULL;
    net->fc1.d_bias    = NULL;
}


void backward_alexnet_16(alexnet *net, const int *batch_Y,
                          const _Float16 *batch_targets, float *loss_out)
{
    calloc_alexnet_d_params_16(net);

    if (net->batchsize > ALEXNET_STATIC_MAX_BATCH) {
        printf_("Error: batchsize %d exceeds static max batch %d\n",
                net->batchsize, ALEXNET_STATIC_MAX_BATCH);
        exit(1);
    }

    _Float16 *curr_grad = d_fc1_output_buf;
    _Float16 *next_grad = d_grad_ping_0;
    float     loss_val  = 0.0f;
    int64_t   t0        = 0;

    t0 = alexnet_cycle_count_local();
#if defined(SET_CEL)
    loss_val = cross_entropy_loss_16(curr_grad, net->fc1.output, batch_Y,
                                     net->fc1.out_units, net->batchsize);
#else
    loss_val = mse_loss_vec_16(curr_grad, net->fc1.output, batch_targets,
                               net->fc1.out_units, net->batchsize);
#endif
    last_loss_cycles = alexnet_cycle_count_local() - t0;

    net->fc1.d_input = next_grad;
    t0 = alexnet_cycle_count_local();
    zero_f16_vec(net->fc1.d_input, net->batchsize * net->fc1.in_units);
    last_zero_dinput_cycles = alexnet_cycle_count_local() - t0;
    net->fc1.d_output = curr_grad;

    if (net->trainable.fc1) {
        t0 = alexnet_cycle_count_local();
        fc_op_backward(&(net->fc1), &last_fc_backward_breakdown);
        last_fc_backward_total_cycles = alexnet_cycle_count_local() - t0;
    } else {
        fc_op_backward_input_only(&(net->fc1));
        last_fc_backward_breakdown.d_input_cycles   = 0;
        last_fc_backward_breakdown.d_bias_cycles    = 0;
        last_fc_backward_breakdown.d_weights_cycles = 0;
        last_fc_backward_total_cycles               = 0;
    }

    ALEXNET_LOG_LAYER(" backward (&(net->fc1)) done\n");

    if (loss_out != NULL)
        *loss_out = loss_val;
}


void alexnet_train_16(alexnet *net, int epochs)
{
    if (net->batchsize > ALEXNET_STATIC_MAX_BATCH) {
        printf_("Error: batchsize %d exceeds static max batch %d\n",
                net->batchsize, ALEXNET_STATIC_MAX_BATCH);
        exit(1);
    }

    net->input = train_input_buf;
    int *batch_Y = train_batch_Y_buf;
    int *preds   = train_preds_buf;

    int dataset_count   = FC_TOTAL_SAMPLES;
    int steps_per_epoch = dataset_count / net->batchsize;
    if (dataset_count % net->batchsize) steps_per_epoch++;
    if (steps_per_epoch <= 0) steps_per_epoch = 1;
    if (ALEXNET_MAX_STEPS > 0 && steps_per_epoch > ALEXNET_MAX_STEPS)
        steps_per_epoch = ALEXNET_MAX_STEPS;

    ALEXNET_LOG_LAYER("\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>> training begin >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n");
    for (int e = 0; e < epochs; e++)
    {
        printf_(">>>>>>>>>>>>>>>>>>>> epoch %d >>>>>>>>>>>>>>>>>>>>>>\n", e + 1);
        ALEXNET_LOG_LAYER("============================= epoch %d / %d =============================\n", e+1, epochs);
        for (int b = 0; b < steps_per_epoch; b++)
        {
            float   step_loss               = 0.0f;
            int64_t prep_cycles             = 0;
            int64_t forward_cycles          = 0;
            int64_t pred_metric_cycles      = 0;
            int64_t backward_wrapper_cycles = 0;
            int64_t t0                      = 0;
            ALEXNET_LOG_LAYER("-----------------------------step %d / %d---------------------------------\n", b+1, steps_per_epoch);

            t0 = alexnet_cycle_count_local();
            int sample_offset = (b * net->batchsize) % dataset_count;

            vcopy_f16(train_input_buf,
                      test_inputs_32 + sample_offset * FC_INPUT_UNITS,
                      (size_t)net->batchsize * FC_INPUT_UNITS);

        #if defined(SET_CEL)
            for (int i = 0; i < net->batchsize; i++)
                batch_Y[i] = test_labels_32[sample_offset + i];
        #else
            vcopy_f16(mse_targets_buf,
                      test_targets_32 + sample_offset * FC_OUTPUT_UNITS,
                      (size_t)net->batchsize * FC_OUTPUT_UNITS);
            for (int i = 0; i < net->batchsize; i++)
                batch_Y[i] = argmax(mse_targets_buf + i * FC_OUTPUT_UNITS, FC_OUTPUT_UNITS);
        #endif
            prep_cycles = alexnet_cycle_count_local() - t0;

            t0 = alexnet_cycle_count_local();
            forward_alexnet(net);
            forward_cycles = alexnet_cycle_count_local() - t0;

            t0 = alexnet_cycle_count_local();
            for (int i = 0; i < net->batchsize; i++)
                preds[i] = argmax(net->output + i * net->fc1.out_units, net->fc1.out_units);

#ifdef SHOW_PREDCITION_DETAIL
            printf_("pred[ ");
            for (int i = 0; i < net->batchsize; i++)
                printf_("%d ", preds[i]);
            printf_("]  label[ ");
            for (int i = 0; i < net->batchsize; i++)
                printf_("%d ", batch_Y[i]);
            printf_("]\n");
#endif
            compute_batch_metrics_16(preds, batch_Y, net->batchsize);
            pred_metric_cycles = alexnet_cycle_count_local() - t0;

            t0 = alexnet_cycle_count_local();
            backward_alexnet_16(net, batch_Y, mse_targets_buf, &step_loss);
            backward_wrapper_cycles = alexnet_cycle_count_local() - t0;

            printf_("cycles[epoch %d batch %d/%d]: prep=%ld, forward=%ld, pred+metric=%ld, loss=%ld, zero_d_input=%ld, backward_d_input=%ld, backward_d_bias=%ld, backward_d_weights=%ld, backward_total=%ld, backward_wrapper=%ld\n",
                    e + 1, b + 1, steps_per_epoch,
                    prep_cycles, forward_cycles, pred_metric_cycles,
                    last_loss_cycles, last_zero_dinput_cycles,
                    last_fc_backward_breakdown.d_input_cycles,
                    last_fc_backward_breakdown.d_bias_cycles,
                    last_fc_backward_breakdown.d_weights_cycles,
                    last_fc_backward_total_cycles,
                    backward_wrapper_cycles);
            printf_("epoch %d step %d/%d loss: %.6f\n", e + 1, b + 1, steps_per_epoch, step_loss);
        }
        ALEXNET_LOG_LAYER("============================= epoch %d / %d end =============================\n", e+1, epochs);
    }
    ALEXNET_LOG_LAYER(">>>>>>>>>>>>>>>>>>>>>>>>>>> training end >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n");

    (void)batch_Y;
    (void)preds;
}

void alexnet_test_16(alexnet *net)
{
    if (net->batchsize > ALEXNET_STATIC_MAX_BATCH) {
        printf_("Error: batchsize %d exceeds static max batch %d\n",
                net->batchsize, ALEXNET_STATIC_MAX_BATCH);
        exit(1);
    }

    int *batch_Y = test_batch_Y_buf;
    int *preds   = test_preds_buf;

    int steps = 1;

    printf_(">>>>>>>>>>>>>>>>>>>>>>>>>> start test pass >>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n");

    _Float16 *test_input  = test_input_buf;
    _Float16 *saved_input = net->input;
    net->input = test_input;

    int total_correct = 0;
    int total_seen    = 0;

    for (int b = 0; b < steps; b++)
    {
        int sample_offset = (b * net->batchsize) % FC_TOTAL_SAMPLES;

        vcopy_f16(net->input,
                  test_inputs_32 + sample_offset * FC_INPUT_UNITS,
                  (size_t)net->batchsize * FC_INPUT_UNITS);

#if defined(SET_CEL)
        for (int i = 0; i < net->batchsize; i++)
            batch_Y[i] = test_labels_32[sample_offset + i];
#else
        vcopy_f16(mse_targets_buf,
                  test_targets_32 + sample_offset * FC_OUTPUT_UNITS,
                  (size_t)net->batchsize * FC_OUTPUT_UNITS);
        for (int i = 0; i < net->batchsize; i++)
            batch_Y[i] = argmax(mse_targets_buf + i * FC_OUTPUT_UNITS, FC_OUTPUT_UNITS);
#endif

        forward_alexnet(net);
        printf_("batch %d/%d  forward done\n", b+1, steps);

        for (int i = 0; i < net->batchsize; i++)
            preds[i] = argmax(net->output + i * net->fc1.out_units, net->fc1.out_units);

#ifdef SHOW_PREDCITION_DETAIL
        printf_("pred[ ");
        for (int i = 0; i < net->batchsize; i++)
            printf_("%d ", preds[i]);
        printf_("]  label[ ");
        for (int i = 0; i < net->batchsize; i++)
            printf_("%d ", batch_Y[i]);
        printf_("]\n");
#endif

        printf_("Test batch %d/%d stats\n", b+1, steps);
        compute_batch_metrics_16(preds, batch_Y, net->batchsize);

        for (int i = 0; i < net->batchsize; i++)
            if (preds[i] == batch_Y[i]) total_correct++;
        total_seen += net->batchsize;
    }

    printf_("\n--- Overall test results: %d / %d correct  (accuracy %.4f) ---\n",
           total_correct, total_seen, (float)total_correct / total_seen);
    printf_(">>>>>>>>>>>>>>>>>>>>>>>>>>> test pass end >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n");

    (void)test_input;
    net->input = saved_input;
    (void)batch_Y;
    (void)preds;
}


void compute_batch_metrics_16(const int *preds, const int *labels, int batchsize)
{
    const int classes = FC_OUTPUT_UNITS;

    int correct = 0;
    for (int i = 0; i < batchsize; i++)
        if (preds[i] == labels[i]) correct++;
    float accuracy = (float)correct / batchsize;
    printf_("batch accuracy:  %.4f  (%d / %d correct)\n", accuracy, correct, batchsize);

    int *true_pos  = metrics_true_pos;
    int *false_pos = metrics_false_pos;
    int *false_neg = metrics_false_neg;

    memset(true_pos,  0, classes * sizeof(int));
    memset(false_pos, 0, classes * sizeof(int));
    memset(false_neg, 0, classes * sizeof(int));

    for (int i = 0; i < batchsize; i++) {
        if (preds[i] < 0 || preds[i] >= classes || labels[i] < 0 || labels[i] >= classes) {
            printf_("[WARNING] Invalid data! Pred: %d, Label: %d\n", preds[i], labels[i]);
            continue;
        }

        if (preds[i] == labels[i]) {
            true_pos[labels[i]]++;
        } else {
            false_pos[preds[i]]++;
            false_neg[labels[i]]++;
        }
    }

    float f1_sum      = 0.0f;
    int   class_count = 0;
    for (int c = 0; c < classes; c++) {
        if (true_pos[c] + false_pos[c] + false_neg[c] > 0) {
            float prec = (true_pos[c] + false_pos[c] > 0)
                         ? (float)true_pos[c] / (true_pos[c] + false_pos[c]) : 0.0f;
            float rec  = (true_pos[c] + false_neg[c] > 0)
                         ? (float)true_pos[c] / (true_pos[c] + false_neg[c]) : 0.0f;
            float f1   = (prec + rec > 0.0f)
                         ? 2.0f * prec * rec / (prec + rec) : 0.0f;
            f1_sum += f1;
            class_count++;
        }
    }
    float macro_f1 = (class_count > 0) ? f1_sum / class_count : 0.0f;
    printf_("batch macro F1:  %.4f  (over %d classes)\n", macro_f1, class_count);
}
