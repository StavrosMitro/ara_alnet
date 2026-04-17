//
// File:        train.c
// Description: Implementation of functions related to training
// Author:      Haris Wang
//
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
// #include <assert.h>
#include <string.h>
#include "alexnet.h"
#include "data.h"
#ifdef SPIKE
#include <printf.h>
#elif defined ARA_LINUX
#include <stdio.h>
#else
#include "printf.h"
#endif

#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))

// #define LEARNING_RATE 0.001
#define LEARNING_RATE 0.01

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
#define ALEXNET_STATIC_MAX_BATCH 1
#endif
#endif

static float d_conv1_weights[C1_CHANNELS*IN_CHANNELS*C1_KERNEL_L*C1_KERNEL_L];
static float d_conv2_weights[C2_CHANNELS*C1_CHANNELS*C2_KERNEL_L*C2_KERNEL_L];
static float d_conv3_weights[C3_CHANNELS*C2_CHANNELS*C3_KERNEL_L*C3_KERNEL_L];
static float d_conv4_weights[C4_CHANNELS*C3_CHANNELS*C4_KERNEL_L*C4_KERNEL_L];
static float d_conv5_weights[C5_CHANNELS*C4_CHANNELS*C5_KERNEL_L*C5_KERNEL_L];
static float d_fc1_weights[C5_CHANNELS*FC6_LAYER*POOLING5_L*POOLING5_L];
static float d_fc2_weights[FC6_LAYER*FC7_LAYER];
static float d_fc3_weights[FC7_LAYER*OUT_LAYER];

static float d_conv1_bias[C1_CHANNELS];
static float d_conv2_bias[C2_CHANNELS];
static float d_conv3_bias[C3_CHANNELS];
static float d_conv4_bias[C4_CHANNELS];
static float d_conv5_bias[C5_CHANNELS];
static float d_fc1_bias[FC6_LAYER];
static float d_fc2_bias[FC7_LAYER];
static float d_fc3_bias[OUT_LAYER];

static float d_bn1_gamma[C1_CHANNELS];
static float d_bn2_gamma[C2_CHANNELS];
static float d_bn3_gamma[C3_CHANNELS];
static float d_bn4_gamma[C4_CHANNELS];
static float d_bn5_gamma[C5_CHANNELS];
static float d_bn1_beta[C1_CHANNELS];
static float d_bn2_beta[C2_CHANNELS];
static float d_bn3_beta[C3_CHANNELS];
static float d_bn4_beta[C4_CHANNELS];
static float d_bn5_beta[C5_CHANNELS];

static float d_fc3_output_buf[ALEXNET_STATIC_MAX_BATCH * OUT_LAYER];

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

// Shared ping-pong buffers for inter-layer gradient propagation.
static float d_grad_ping_0[ALEXNET_STATIC_MAX_BATCH * ALEXNET_MAX_BACKWARD_UNITS];
static float d_grad_ping_1[ALEXNET_STATIC_MAX_BATCH * ALEXNET_MAX_BACKWARD_UNITS];

static float train_input_buf[ALEXNET_STATIC_MAX_BATCH * IN_CHANNELS * FEATURE0_L * FEATURE0_L];
static int train_batch_Y_buf[ALEXNET_STATIC_MAX_BATCH];
static int train_preds_buf[ALEXNET_STATIC_MAX_BATCH];

static float test_input_buf[ALEXNET_STATIC_MAX_BATCH * IN_CHANNELS * FEATURE0_L * FEATURE0_L];
static int test_batch_Y_buf[ALEXNET_STATIC_MAX_BATCH];
static int test_preds_buf[ALEXNET_STATIC_MAX_BATCH];

static void zero_f32(float *buf, int n)
{
    memset(buf, 0, (size_t)n * sizeof(float));
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


static void cross_entropy_loss(float *delta_preds, const float *preds, const int *labels, int units, int BATCH_SIZE)
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
}


static float v_conv1_weights[C1_CHANNELS*IN_CHANNELS*C1_KERNEL_L*C1_KERNEL_L];
static float v_conv2_weights[C2_CHANNELS*C1_CHANNELS*C2_KERNEL_L*C2_KERNEL_L];
static float v_conv3_weights[C3_CHANNELS*C2_CHANNELS*C3_KERNEL_L*C3_KERNEL_L];
static float v_conv4_weights[C4_CHANNELS*C3_CHANNELS*C4_KERNEL_L*C4_KERNEL_L];
static float v_conv5_weights[C5_CHANNELS*C4_CHANNELS*C5_KERNEL_L*C5_KERNEL_L];
static float v_fc1_weights[C5_CHANNELS*FC6_LAYER*POOLING5_L*POOLING5_L];
static float v_fc2_weights[FC6_LAYER*FC7_LAYER];
static float v_fc3_weights[FC7_LAYER*OUT_LAYER];

static float v_conv1_bias[C1_CHANNELS];
static float v_conv2_bias[C2_CHANNELS];
static float v_conv3_bias[C3_CHANNELS];
static float v_conv4_bias[C4_CHANNELS];
static float v_conv5_bias[C5_CHANNELS];
static float v_fc1_bias[FC6_LAYER];
static float v_fc2_bias[FC7_LAYER];
static float v_fc3_bias[OUT_LAYER];

static float v_bn1_gamma[C1_CHANNELS];
static float v_bn2_gamma[C2_CHANNELS];
static float v_bn3_gamma[C3_CHANNELS];
static float v_bn4_gamma[C4_CHANNELS];
static float v_bn5_gamma[C5_CHANNELS];
static float v_bn1_beta[C1_CHANNELS];
static float v_bn2_beta[C2_CHANNELS];
static float v_bn3_beta[C3_CHANNELS];
static float v_bn4_beta[C4_CHANNELS];
static float v_bn5_beta[C5_CHANNELS];


static inline void CLIP(float *x, float down, float up)
{
    *x = MIN(up, MAX(down, *x));
}

static void momentum_sgd(float *w, float *v_w, float *d_w, int units)
{
    /**
     * momentum stochastic gradient descent
     * 
     * Input:
     *      w   [units]
     *      v_w [units]
     *      d_w [units]
     * Output:
     *      w   [units]
     *      v_w [units]
     * */     
    for (int i = 0; i < units; i++)
    {
        v_w[i] = 0.9 * v_w[i] - LEARNING_RATE * d_w[i];
        CLIP(v_w+i, -1, 1);
        w[i] = w[i] + v_w[i];
    }
}


static void gradient_descent_a(void *argv)
{
    alexnet *net = (alexnet *)argv;
    if (net->trainable.fc1)
        momentum_sgd(net->fc1.weights, v_fc1_weights, net->fc1.d_weights, C5_CHANNELS*FC6_LAYER*POOLING5_L*POOLING5_L);
}

static void gradient_descent_b(void *argv)
{
    alexnet *net = (alexnet *)argv;
    if (net->trainable.fc2)
        momentum_sgd(net->fc2.weights, v_fc2_weights, net->fc2.d_weights, FC6_LAYER*FC7_LAYER);
}

static void gradient_descent_c(void *argv)
{
    alexnet *net = (alexnet *)argv;
    if (net->trainable.fc3)
        momentum_sgd(net->fc3.weights, v_fc3_weights, net->fc3.d_weights, FC7_LAYER*OUT_LAYER);
}

static void gradient_descent_d(void *argv)
{
    alexnet *net = (alexnet *)argv;

    if (net->trainable.conv1)
        momentum_sgd(net->conv1.weights, v_conv1_weights, net->conv1.d_weights, C1_CHANNELS*IN_CHANNELS*C1_KERNEL_L*C1_KERNEL_L);
    if (net->trainable.conv2)
        momentum_sgd(net->conv2.weights, v_conv2_weights, net->conv2.d_weights, C2_CHANNELS*C1_CHANNELS*C2_KERNEL_L*C2_KERNEL_L);
    if (net->trainable.conv3)
        momentum_sgd(net->conv3.weights, v_conv3_weights, net->conv3.d_weights, C3_CHANNELS*C2_CHANNELS*C3_KERNEL_L*C3_KERNEL_L);
    if (net->trainable.conv4)
        momentum_sgd(net->conv4.weights, v_conv4_weights, net->conv4.d_weights, C4_CHANNELS*C3_CHANNELS*C4_KERNEL_L*C4_KERNEL_L);
    if (net->trainable.conv5)
        momentum_sgd(net->conv5.weights, v_conv5_weights, net->conv5.d_weights, C5_CHANNELS*C4_CHANNELS*C5_KERNEL_L*C5_KERNEL_L);

    if (net->trainable.conv1)
        momentum_sgd(net->conv1.bias,   v_conv1_bias,   net->conv1.d_bias, C1_CHANNELS);
    if (net->trainable.conv2)
        momentum_sgd(net->conv2.bias,   v_conv2_bias,   net->conv2.d_bias, C2_CHANNELS);
    if (net->trainable.conv3)
        momentum_sgd(net->conv3.bias,   v_conv3_bias,   net->conv3.d_bias, C3_CHANNELS);
    if (net->trainable.conv4)
        momentum_sgd(net->conv4.bias,   v_conv4_bias,   net->conv4.d_bias, C4_CHANNELS);
    if (net->trainable.conv5)
        momentum_sgd(net->conv5.bias,   v_conv5_bias,   net->conv5.d_bias, C5_CHANNELS);
    if (net->trainable.fc1)
        momentum_sgd(net->fc1.bias,     v_fc1_bias,     net->fc1.d_bias,   FC6_LAYER);
    if (net->trainable.fc2)
        momentum_sgd(net->fc2.bias,     v_fc2_bias,     net->fc2.d_bias,   FC7_LAYER);
    if (net->trainable.fc3)
        momentum_sgd(net->fc3.bias,     v_fc3_bias,     net->fc3.d_bias,   OUT_LAYER);

    if (net->trainable.bn1) {
        momentum_sgd(net->bn1.gamma, v_bn1_gamma, net->bn1.d_gamma, C1_CHANNELS);
        momentum_sgd(net->bn1.beta,  v_bn1_beta,  net->bn1.d_beta,  C1_CHANNELS);
    }
    if (net->trainable.bn2) {
        momentum_sgd(net->bn2.gamma, v_bn2_gamma, net->bn2.d_gamma, C2_CHANNELS);
        momentum_sgd(net->bn2.beta,  v_bn2_beta,  net->bn2.d_beta,  C2_CHANNELS);
    }
    if (net->trainable.bn3) {
        momentum_sgd(net->bn3.gamma, v_bn3_gamma, net->bn3.d_gamma, C3_CHANNELS);
        momentum_sgd(net->bn3.beta,  v_bn3_beta,  net->bn3.d_beta,  C3_CHANNELS);
    }
    if (net->trainable.bn4) {
        momentum_sgd(net->bn4.gamma, v_bn4_gamma, net->bn4.d_gamma, C4_CHANNELS);
        momentum_sgd(net->bn4.beta,  v_bn4_beta,  net->bn4.d_beta,  C4_CHANNELS);
    }
    if (net->trainable.bn5) {
        momentum_sgd(net->bn5.gamma, v_bn5_gamma, net->bn5.d_gamma, C5_CHANNELS);
        momentum_sgd(net->bn5.beta,  v_bn5_beta,  net->bn5.d_beta,  C5_CHANNELS);
    }
}

static void gradient_descent(alexnet *net)
{
    gradient_descent_a((void *)(net));
    gradient_descent_b((void *)(net));
    gradient_descent_c((void *)(net));
    gradient_descent_d((void *)(net));
}


void calloc_alexnet_d_params(alexnet *net)
{
    net->conv1.d_weights = NULL;
    net->conv2.d_weights = NULL;
    net->conv3.d_weights = NULL;
    net->conv4.d_weights = NULL;
    net->conv5.d_weights = d_conv5_weights;
    net->fc1.d_weights = NULL;
    net->fc2.d_weights = d_fc2_weights;
    net->fc3.d_weights = d_fc3_weights;

    net->conv1.d_bias = NULL;
    net->conv2.d_bias = NULL;
    net->conv3.d_bias = NULL;
    net->conv4.d_bias = NULL;
    net->conv5.d_bias = d_conv5_bias;
    net->fc1.d_bias = NULL;
    net->fc2.d_bias = d_fc2_bias;
    net->fc3.d_bias = d_fc3_bias;

    net->bn1.d_gamma = d_bn1_gamma;
    net->bn2.d_gamma = d_bn2_gamma;
    net->bn3.d_gamma = d_bn3_gamma;
    net->bn4.d_gamma = d_bn4_gamma;
    net->bn5.d_gamma = d_bn5_gamma;
    net->bn1.d_beta = d_bn1_beta;
    net->bn2.d_beta = d_bn2_beta;
    net->bn3.d_beta = d_bn3_beta;
    net->bn4.d_beta = d_bn4_beta;
    net->bn5.d_beta = d_bn5_beta;

    zero_f32(net->conv5.d_weights, C5_CHANNELS*C4_CHANNELS*C5_KERNEL_L*C5_KERNEL_L);
    zero_f32(net->fc2.d_weights, FC6_LAYER*FC7_LAYER);
    zero_f32(net->fc3.d_weights, FC7_LAYER*OUT_LAYER);

    zero_f32(net->conv5.d_bias, C5_CHANNELS);
    zero_f32(net->fc2.d_bias, FC7_LAYER);
    zero_f32(net->fc3.d_bias, OUT_LAYER);

    zero_f32(net->bn1.d_gamma, C1_CHANNELS);
    zero_f32(net->bn2.d_gamma, C2_CHANNELS);
    zero_f32(net->bn3.d_gamma, C3_CHANNELS);
    zero_f32(net->bn4.d_gamma, C4_CHANNELS);
    zero_f32(net->bn5.d_gamma, C5_CHANNELS);
    zero_f32(net->bn1.d_beta, C1_CHANNELS);
    zero_f32(net->bn2.d_beta, C2_CHANNELS);
    zero_f32(net->bn3.d_beta, C3_CHANNELS);
    zero_f32(net->bn4.d_beta, C4_CHANNELS);
    zero_f32(net->bn5.d_beta, C5_CHANNELS);
}

void free_alexnet_d_params(alexnet *net)
{
    net->conv1.d_weights = NULL;
    net->conv2.d_weights = NULL;
    net->conv3.d_weights = NULL;
    net->conv4.d_weights = NULL;
    net->conv5.d_weights = NULL;
    net->fc1.d_weights = NULL;
    net->fc2.d_weights = NULL;
    net->fc3.d_weights = NULL;

    net->conv1.d_bias = NULL;
    net->conv2.d_bias = NULL;
    net->conv3.d_bias = NULL;
    net->conv4.d_bias = NULL;
    net->conv5.d_bias = NULL;
    net->fc1.d_bias = NULL;
    net->fc2.d_bias = NULL;
    net->fc3.d_bias = NULL;

    net->bn1.d_gamma = NULL;
    net->bn2.d_gamma = NULL;
    net->bn3.d_gamma = NULL;
    net->bn4.d_gamma = NULL;
    net->bn5.d_gamma = NULL;
    net->bn1.d_beta = NULL;
    net->bn2.d_beta = NULL;
    net->bn3.d_beta = NULL;
    net->bn4.d_beta = NULL;
    net->bn5.d_beta = NULL;
}

void backward_alexnet(alexnet *net, int *batch_Y)
{
    /**
     * alexnet backward
     * 
     * Input:
     *      net:      our network
     *      batch_Y:  labels of images
     * Output:
     *      net
     * */
    calloc_alexnet_d_params(net);

    if (net->batchsize > ALEXNET_STATIC_MAX_BATCH) {
        printf_("Error: batchsize %d exceeds static max batch %d\n", net->batchsize, ALEXNET_STATIC_MAX_BATCH);
        exit(1);
    }

    net->fc3.d_output = d_fc3_output_buf;
    zero_f32(net->fc3.d_output, net->batchsize * net->fc3.out_units);
    cross_entropy_loss(net->fc3.d_output, net->output, batch_Y, OUT_LAYER, net->fc3.batchsize);

    float *curr_grad = NULL;
    float *next_grad = d_grad_ping_0;

#ifdef SHOW_OP_TIME
    alexnet_timer_t start = {0};
    alexnet_timer_t finish = {0};
    double duration = 0.0;
#endif

#ifdef SHOW_OP_TIME
    alexnet_timer_now(&start);
#endif
    net->fc3.d_input = next_grad;
    zero_f32(net->fc3.d_input, net->batchsize * net->fc3.in_units);
    if (net->trainable.fc3)
        fc_op_backward_full(&(net->fc3));
    else
        fc_op_backward_input_only(&(net->fc3));
    ALEXNET_LOG_LAYER(" backward (&(net->fc3)) done\n");
    curr_grad = net->fc3.d_input;
    next_grad = (next_grad == d_grad_ping_0) ? d_grad_ping_1 : d_grad_ping_0;
#ifdef SHOW_OP_TIME
    alexnet_timer_now(&finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
#endif

    net->relu7.d_input = next_grad;
    zero_f32(net->relu7.d_input, net->batchsize * net->relu7.units);
    net->relu7.d_output = curr_grad;
    relu_op_backward(&(net->relu7));
    curr_grad = net->relu7.d_input;
    next_grad = (next_grad == d_grad_ping_0) ? d_grad_ping_1 : d_grad_ping_0;

#ifdef SHOW_OP_TIME
    alexnet_timer_now(&start);
#endif
    net->fc2.d_input = next_grad;
    zero_f32(net->fc2.d_input, net->batchsize * net->fc2.in_units);
    net->fc2.d_output = curr_grad;
    if (net->trainable.fc2)
        fc_op_backward_full(&(net->fc2));
    else
        fc_op_backward_input_only(&(net->fc2));
    ALEXNET_LOG_LAYER(" backward (&(net->fc2)) done\n");
    curr_grad = net->fc2.d_input;
    next_grad = (next_grad == d_grad_ping_0) ? d_grad_ping_1 : d_grad_ping_0;
#ifdef SHOW_OP_TIME
    alexnet_timer_now(&finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
#endif

    net->relu6.d_input = next_grad;
    zero_f32(net->relu6.d_input, net->batchsize * net->relu6.units);
    net->relu6.d_output = curr_grad;
    relu_op_backward(&(net->relu6));
    curr_grad = net->relu6.d_input;
    next_grad = (next_grad == d_grad_ping_0) ? d_grad_ping_1 : d_grad_ping_0;

#ifdef SHOW_OP_TIME
    alexnet_timer_now(&start);
#endif
    net->fc1.d_input = next_grad;
    zero_f32(net->fc1.d_input, net->batchsize * net->fc1.in_units);
    net->fc1.d_output = curr_grad;
    if (net->trainable.fc1)
        fc_op_backward_full(&(net->fc1));
    else
        fc_op_backward_input_only(&(net->fc1));
    ALEXNET_LOG_LAYER(" backward (&(net->fc1)) done\n");
    curr_grad = net->fc1.d_input;
    next_grad = (next_grad == d_grad_ping_0) ? d_grad_ping_1 : d_grad_ping_0;
#ifdef SHOW_OP_TIME
    alexnet_timer_now(&finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
#endif

#ifdef SHOW_OP_TIME
    alexnet_timer_now(&start);
#endif
    net->mp5.d_input = next_grad;
    zero_f32(net->mp5.d_input, net->batchsize * net->mp5.in_units);
    net->mp5.d_output = curr_grad;
    max_pooling_op_backward(&(net->mp5));
    ALEXNET_LOG_LAYER(" backward (&(net->mp5)) done\n");
    curr_grad = net->mp5.d_input;
    next_grad = (next_grad == d_grad_ping_0) ? d_grad_ping_1 : d_grad_ping_0;
#ifdef SHOW_OP_TIME
    alexnet_timer_now(&finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
#endif

    net->relu5.d_input = next_grad;
    zero_f32(net->relu5.d_input, net->batchsize * net->relu5.units);
    net->relu5.d_output = curr_grad;
    relu_op_backward(&(net->relu5));
    curr_grad = net->relu5.d_input;
    next_grad = (next_grad == d_grad_ping_0) ? d_grad_ping_1 : d_grad_ping_0;

#ifdef SHOW_OP_TIME
    alexnet_timer_now(&start);
#endif
    net->bn5.d_input = next_grad;
    zero_f32(net->bn5.d_input, net->batchsize * net->bn5.units);
    net->bn5.d_output = curr_grad;
    if (net->trainable.bn5)
        batch_norm_op_backward_full(&(net->bn5));
    else
        batch_norm_op_backward_input_only(&(net->bn5));
    ALEXNET_LOG_LAYER(" backward (&(net->bn5)) done\n");
    curr_grad = net->bn5.d_input;
    next_grad = (next_grad == d_grad_ping_0) ? d_grad_ping_1 : d_grad_ping_0;
    for(int i=0; i< net->batchsize * net->bn5.units; i++)
    {
        if(net->bn5.d_input[i] > 4)
        {
            printf_("!!!!!!!!!!!!!!!!!!!!!!!!!!! bn5 threshold hit at idx %d\n", i);
            break;
        }
    }

#ifdef SHOW_OP_TIME
    alexnet_timer_now(&finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
#endif

#ifdef SHOW_OP_TIME
    alexnet_timer_now(&start);
#endif
    net->conv5.d_input = next_grad;
    zero_f32(net->conv5.d_input, net->batchsize * net->conv5.in_units);
    net->conv5.d_output = curr_grad;
    if (net->trainable.conv5)
        conv_op_backward_full(&(net->conv5));
    else
        conv_op_backward_input_only(&(net->conv5));
    ALEXNET_LOG_LAYER(" backward (&(net->conv5)) done\n");
    curr_grad = net->conv5.d_input;
    next_grad = (next_grad == d_grad_ping_0) ? d_grad_ping_1 : d_grad_ping_0;
#ifdef SHOW_OP_TIME
    alexnet_timer_now(&finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
#endif

    net->relu4.d_input = next_grad;
    zero_f32(net->relu4.d_input, net->batchsize * net->relu4.units);
    net->relu4.d_output = curr_grad;
    relu_op_backward(&(net->relu4));
    curr_grad = net->relu4.d_input;
    next_grad = (next_grad == d_grad_ping_0) ? d_grad_ping_1 : d_grad_ping_0;

#ifdef SHOW_OP_TIME
    alexnet_timer_now(&start);
#endif
    net->bn4.d_input = next_grad;
    zero_f32(net->bn4.d_input, net->batchsize * net->bn4.units);
    net->bn4.d_output = curr_grad;
    if (net->trainable.bn4)
        batch_norm_op_backward_full(&(net->bn4));
    else
        batch_norm_op_backward_input_only(&(net->bn4));
    ALEXNET_LOG_LAYER(" backward (&(net->bn4)) done\n");
    curr_grad = net->bn4.d_input;
    next_grad = (next_grad == d_grad_ping_0) ? d_grad_ping_1 : d_grad_ping_0;
#ifdef SHOW_OP_TIME
    alexnet_timer_now(&finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
#endif

    for(int i=0; i< net->batchsize * net->bn4.units; i++)
    {
        if(net->bn4.d_input[i] > 4)
        {
            ALEXNET_LOG_LAYER("!!!!!!!!!!!!!!!!!!!!!!!!!!! bn4 threshold hit at idx %d\n", i);
            break;
        }
    }

#ifdef SHOW_OP_TIME
    alexnet_timer_now(&start);
#endif
    net->conv4.d_input = next_grad;
    zero_f32(net->conv4.d_input, net->batchsize * net->conv4.in_units);
    net->conv4.d_output = curr_grad;
    if (net->trainable.conv4)
        conv_op_backward_full(&(net->conv4));
    else
        conv_op_backward_input_only(&(net->conv4));
    ALEXNET_LOG_LAYER(" backward (&(net->conv4)) done\n");
    curr_grad = net->conv4.d_input;
    next_grad = (next_grad == d_grad_ping_0) ? d_grad_ping_1 : d_grad_ping_0;
#ifdef SHOW_OP_TIME
    alexnet_timer_now(&finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
#endif

    net->relu3.d_input = next_grad;
    zero_f32(net->relu3.d_input, net->batchsize * net->relu3.units);
    net->relu3.d_output = curr_grad;
    relu_op_backward(&(net->relu3));
    curr_grad = net->relu3.d_input;
    next_grad = (next_grad == d_grad_ping_0) ? d_grad_ping_1 : d_grad_ping_0;

#ifdef SHOW_OP_TIME
    alexnet_timer_now(&start);
#endif
    net->bn3.d_input = next_grad;
    zero_f32(net->bn3.d_input, net->batchsize * net->bn3.units);
    net->bn3.d_output = curr_grad;
    if (net->trainable.bn3)
        batch_norm_op_backward_full(&(net->bn3));
    else
        batch_norm_op_backward_input_only(&(net->bn3));
    ALEXNET_LOG_LAYER(" backward (&(net->bn3)) done\n");
    curr_grad = net->bn3.d_input;
    next_grad = (next_grad == d_grad_ping_0) ? d_grad_ping_1 : d_grad_ping_0;
#ifdef SHOW_OP_TIME
    alexnet_timer_now(&finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
#endif

    for(int i=0; i< net->batchsize * net->bn3.units; i++)
    {
        if(net->bn3.d_input[i] > 4)
        {
            ALEXNET_LOG_LAYER("!!!!!!!!!!!!!!!!!!!!!!!!!!! bn3 threshold hit at idx %d\n", i);
            break;
        }
    }

#ifdef SHOW_OP_TIME
    alexnet_timer_now(&start);
#endif
    net->conv3.d_input = next_grad;
    zero_f32(net->conv3.d_input, net->batchsize * net->conv3.in_units);
    net->conv3.d_output = curr_grad;
    if (net->trainable.conv3)
        conv_op_backward_full(&(net->conv3));
    else
        conv_op_backward_input_only(&(net->conv3));
    ALEXNET_LOG_LAYER(" backward (&(net->conv3)) done\n");
    curr_grad = net->conv3.d_input;
    next_grad = (next_grad == d_grad_ping_0) ? d_grad_ping_1 : d_grad_ping_0;
#ifdef SHOW_OP_TIME
    alexnet_timer_now(&finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
#endif

    for(int i=0; i< net->batchsize * net->conv3.in_units; i++)
    {
        if(net->conv3.d_input[i] > 4)
        {
            ALEXNET_LOG_LAYER("!!!!!!!!!!!!!!!!!!!!!!!!!!! conv3 threshold hit at idx %d\n", i);
            break;
        }
    }

#ifdef SHOW_OP_TIME
    alexnet_timer_now(&start);
#endif
    net->mp2.d_input = next_grad;
    zero_f32(net->mp2.d_input, net->batchsize * net->mp2.in_units);
    net->mp2.d_output = curr_grad;
    max_pooling_op_backward(&(net->mp2));
    ALEXNET_LOG_LAYER(" backward (&(net->mp2)) done\n");
    curr_grad = net->mp2.d_input;
    next_grad = (next_grad == d_grad_ping_0) ? d_grad_ping_1 : d_grad_ping_0;
#ifdef SHOW_OP_TIME
    alexnet_timer_now(&finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
#endif

    net->relu2.d_input = next_grad;
    zero_f32(net->relu2.d_input, net->batchsize * net->relu2.units);
    net->relu2.d_output = curr_grad;
    relu_op_backward(&(net->relu2));
    curr_grad = net->relu2.d_input;
    next_grad = (next_grad == d_grad_ping_0) ? d_grad_ping_1 : d_grad_ping_0;

#ifdef SHOW_OP_TIME
    alexnet_timer_now(&start);
#endif
    net->bn2.d_input = next_grad;
    zero_f32(net->bn2.d_input, net->batchsize * net->bn2.units);
    net->bn2.d_output = curr_grad;
    if (net->trainable.bn2)
        batch_norm_op_backward_full(&(net->bn2));
    else
        batch_norm_op_backward_input_only(&(net->bn2));
    ALEXNET_LOG_LAYER(" backward (&(net->bn2)) done\n");
    curr_grad = net->bn2.d_input;
    next_grad = (next_grad == d_grad_ping_0) ? d_grad_ping_1 : d_grad_ping_0;
#ifdef SHOW_OP_TIME
    alexnet_timer_now(&finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
#endif

    for(int i=0; i< net->batchsize * net->bn2.units; i++)
    {
        if(net->bn2.d_input[i] > 4)
        {
            ALEXNET_LOG_LAYER("!!!!!!!!!!!!!!!!!!!!!!!!!!! bn2 threshold hit at idx %d\n", i);
            break;
        }
    }

#ifdef SHOW_OP_TIME
    alexnet_timer_now(&start);
#endif
    net->conv2.d_input = next_grad;
    zero_f32(net->conv2.d_input, net->batchsize * net->conv2.in_units);
    net->conv2.d_output = curr_grad;
    if (net->trainable.conv2)
        conv_op_backward_full(&(net->conv2));
    else
        conv_op_backward_input_only(&(net->conv2));
    ALEXNET_LOG_LAYER(" backward (&(net->conv2)) done\n");
    curr_grad = net->conv2.d_input;
    next_grad = (next_grad == d_grad_ping_0) ? d_grad_ping_1 : d_grad_ping_0;
#ifdef SHOW_OP_TIME
    alexnet_timer_now(&finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
#endif

#ifdef SHOW_OP_TIME
    alexnet_timer_now(&start);
#endif
    net->mp1.d_input = next_grad;
    zero_f32(net->mp1.d_input, net->batchsize * net->mp1.in_units);
    net->mp1.d_output = curr_grad;
    max_pooling_op_backward(&(net->mp1));
    ALEXNET_LOG_LAYER(" backward (&(net->mp1)) done\n");
    curr_grad = net->mp1.d_input;
    next_grad = (next_grad == d_grad_ping_0) ? d_grad_ping_1 : d_grad_ping_0;
#ifdef SHOW_OP_TIME
    alexnet_timer_now(&finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
#endif

    net->relu1.d_input = next_grad;
    zero_f32(net->relu1.d_input, net->batchsize * net->relu1.units);
    net->relu1.d_output = curr_grad;
    relu_op_backward(&(net->relu1));
    curr_grad = net->relu1.d_input;
    next_grad = (next_grad == d_grad_ping_0) ? d_grad_ping_1 : d_grad_ping_0;

#ifdef SHOW_OP_TIME
    alexnet_timer_now(&start);
#endif
    net->bn1.d_input = next_grad;
    zero_f32(net->bn1.d_input, net->batchsize * net->bn1.units);
    net->bn1.d_output = curr_grad;
    if (net->trainable.bn1)
        batch_norm_op_backward_full(&(net->bn1));
    else
        batch_norm_op_backward_input_only(&(net->bn1));
    ALEXNET_LOG_LAYER(" backward (&(net->bn1)) done\n");
    curr_grad = net->bn1.d_input;
    next_grad = (next_grad == d_grad_ping_0) ? d_grad_ping_1 : d_grad_ping_0;
#ifdef SHOW_OP_TIME
    alexnet_timer_now(&finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
#endif

    for(int i=0; i< net->batchsize * net->bn1.units; i++)
    {
        if(net->bn1.d_input[i] > 4)
        {
            ALEXNET_LOG_LAYER("!!!!!!!!!!!!!!!!!!!!!!!!!!! bn1 threshold hit at idx %d\n", i);
            break;
        }
    }

#ifdef SHOW_OP_TIME
    alexnet_timer_now(&start);
#endif
    net->conv1.d_input = next_grad;
    zero_f32(net->conv1.d_input, net->batchsize * net->conv1.in_units);
    net->conv1.d_output = curr_grad;
    if (net->trainable.conv1)
        conv_op_backward_full(&(net->conv1));
    else
        conv_op_backward_input_only(&(net->conv1));
    ALEXNET_LOG_LAYER(" backward (&(net->conv1)) done\n");
    curr_grad = net->conv1.d_input;
#ifdef SHOW_OP_TIME
    alexnet_timer_now(&finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
#endif

#ifdef SHOW_OP_TIME
    alexnet_timer_now(&start);
#endif
    gradient_descent(net);
    ALEXNET_LOG_LAYER(" backward update_params(net) done\n");
#ifdef SHOW_OP_TIME
    alexnet_timer_now(&finish);
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
#endif

    free_alexnet_d_params(net);
}


void alexnet_train(alexnet *net, int epochs)
{
    if (net->batchsize > ALEXNET_STATIC_MAX_BATCH) {
        printf_("Error: batchsize %d exceeds static max batch %d\n", net->batchsize, ALEXNET_STATIC_MAX_BATCH);
        exit(1);
    }

    net->input = train_input_buf;
    int *batch_Y = train_batch_Y_buf;
    int *preds = train_preds_buf;

    int dataset_count = get_dataset_count();
    int steps_per_epoch = dataset_count / net->batchsize;
    if (dataset_count % net->batchsize) steps_per_epoch++;
    if (steps_per_epoch <= 0) steps_per_epoch = 1;
    if (ALEXNET_MAX_STEPS > 0 && steps_per_epoch > ALEXNET_MAX_STEPS)
        steps_per_epoch = ALEXNET_MAX_STEPS;

    ALEXNET_LOG_LAYER("\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>> training begin >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n");
    for (int e = 0; e < epochs; e++)
    {
        ALEXNET_LOG_LAYER("============================= epoch %d / %d =============================\n", e+1, epochs);
        for (int b = 0; b < steps_per_epoch; b++)
        {
            ALEXNET_LOG_LAYER("-----------------------------step %d / %d---------------------------------\n", b+1, steps_per_epoch);

            // get_same_batch(net->batchsize, net->input, batch_Y,
            //                net->conv1.in_w, net->conv1.in_h, net->conv1.in_channels, net->fc3.out_units);

            get_next_batch(net->batchsize, net->input, batch_Y,
                           net->conv1.in_w, net->conv1.in_h, net->conv1.in_channels, net->fc3.out_units);

            forward_alexnet(net);

            for (int i = 0; i < net->batchsize; i++)
                preds[i] = argmax(net->output + i * net->fc3.out_units, net->fc3.out_units);

#ifdef SHOW_PREDCITION_DETAIL
            printf_("pred[ ");
            for (int i = 0; i < net->batchsize; i++)
                printf_("%d ", preds[i]);
            printf_("]  label[ ");
            for (int i = 0; i < net->batchsize; i++)
                printf_("%d ", batch_Y[i]);
            printf_("]\n");
#endif
            compute_batch_metrics(preds, batch_Y, net->batchsize);

            backward_alexnet(net, batch_Y);
        }
        ALEXNET_LOG_LAYER("============================= epoch %d / %d end =============================\n", e+1, epochs);
    }
    ALEXNET_LOG_LAYER(">>>>>>>>>>>>>>>>>>>>>>>>>>> training end >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n");

    (void)batch_Y;
    (void)preds;
}

void alexnet_test(alexnet *net)
{
    if (net->batchsize > ALEXNET_STATIC_MAX_BATCH) {
        printf_("Error: batchsize %d exceeds static max batch %d\n", net->batchsize, ALEXNET_STATIC_MAX_BATCH);
        exit(1);
    }

    int *batch_Y = test_batch_Y_buf;
    int *preds   = test_preds_buf;

    // In-memory one-batch evaluation (same data source path as training).
    int steps = 1;

    printf_(">>>>>>>>>>>>>>>>>>>>>>>>>> start test pass >>>>>>>>>>>>>>>>>>>>>>>>>>>>\n\n");

    // allocate a fresh input buffer for the test pass
    float *test_input = test_input_buf;
    float *saved_input = net->input;
    net->input = test_input;

    int total_correct = 0;
    int total_seen    = 0;

    for (int b = 0; b < steps; b++)
    {
        get_next_batch(net->batchsize, net->input, batch_Y,
                       net->conv1.in_w, net->conv1.in_h, net->conv1.in_channels, net->fc3.out_units);

        forward_alexnet(net);
        printf_("batch %d/%d  forward done\n", b+1, steps);

        for (int i = 0; i < net->batchsize; i++)
            preds[i] = argmax(net->output + i * net->fc3.out_units, net->fc3.out_units);

        free_forward_activations(net);   // release all layer outputs allocated by forward

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
        compute_batch_metrics(preds, batch_Y, net->batchsize);

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


void compute_batch_metrics(const int *preds, const int *labels, int batchsize)
{
    // Batch accuracy
    int correct = 0;
    for (int i = 0; i < batchsize; i++)
        if (preds[i] == labels[i]) correct++;
    float accuracy = (float)correct / batchsize;
    printf_("batch accuracy:  %.4f  (%d / %d correct)\n", accuracy, correct, batchsize);

    // Macro-averaged F1 score over classes present in this batch
    int true_pos[OUT_LAYER], false_pos[OUT_LAYER], false_neg[OUT_LAYER];
    memset(true_pos,  0, OUT_LAYER * sizeof(int));
    memset(false_pos, 0, OUT_LAYER * sizeof(int));
    memset(false_neg, 0, OUT_LAYER * sizeof(int));
    for (int i = 0; i < batchsize; i++) {
        if (preds[i] == labels[i]) {
            true_pos[labels[i]]++;
        } else {
            false_pos[preds[i]]++;
            false_neg[labels[i]]++;
        }
    }
    float f1_sum = 0.0f;
    int   class_count = 0;
    for (int c = 0; c < OUT_LAYER; c++) {
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
