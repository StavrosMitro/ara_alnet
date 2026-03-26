//
// File:        alexnet.c
// Description: alexnet.c
// Author:      Haris Wang
//
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <stdint.h>
#include "kernel/alexnet.h"
#include "kernel/weights.h"
#include "kernel/image_inference.h"
#include "kernel/cifar10_dataset.h"
#include "printf.h"

#if !defined(ALEXNET_MODE_TRAIN) && !defined(ALEXNET_MODE_INFERENCE)
#define ALEXNET_MODE_TRAIN
#endif

#ifndef ALEXNET_STATIC_MAX_BATCH
#ifdef ALEXNET_BATCHSIZE
#define ALEXNET_STATIC_MAX_BATCH ALEXNET_BATCHSIZE
#else
#define ALEXNET_STATIC_MAX_BATCH 400
#endif
#endif

static int metrics_totPred[OUT_LAYER];
static int metrics_totLabel[OUT_LAYER];
static int metrics_TP[OUT_LAYER];

static float act_conv1[ALEXNET_STATIC_MAX_BATCH * C1_CHANNELS * FEATURE1_L * FEATURE1_L];
static float act_bn1[ALEXNET_STATIC_MAX_BATCH * C1_CHANNELS * FEATURE1_L * FEATURE1_L];
static float act_relu1[ALEXNET_STATIC_MAX_BATCH * C1_CHANNELS * FEATURE1_L * FEATURE1_L];
static float act_mp1[ALEXNET_STATIC_MAX_BATCH * C1_CHANNELS * POOLING1_L * POOLING1_L];

static float act_conv2[ALEXNET_STATIC_MAX_BATCH * C2_CHANNELS * FEATURE2_L * FEATURE2_L];
static float act_bn2[ALEXNET_STATIC_MAX_BATCH * C2_CHANNELS * FEATURE2_L * FEATURE2_L];
static float act_relu2[ALEXNET_STATIC_MAX_BATCH * C2_CHANNELS * FEATURE2_L * FEATURE2_L];
static float act_mp2[ALEXNET_STATIC_MAX_BATCH * C2_CHANNELS * POOLING2_L * POOLING2_L];

static float act_conv3[ALEXNET_STATIC_MAX_BATCH * C3_CHANNELS * FEATURE3_L * FEATURE3_L];
static float act_bn3[ALEXNET_STATIC_MAX_BATCH * C3_CHANNELS * FEATURE3_L * FEATURE3_L];
static float act_relu3[ALEXNET_STATIC_MAX_BATCH * C3_CHANNELS * FEATURE3_L * FEATURE3_L];

static float act_conv4[ALEXNET_STATIC_MAX_BATCH * C4_CHANNELS * FEATURE4_L * FEATURE4_L];
static float act_bn4[ALEXNET_STATIC_MAX_BATCH * C4_CHANNELS * FEATURE4_L * FEATURE4_L];
static float act_relu4[ALEXNET_STATIC_MAX_BATCH * C4_CHANNELS * FEATURE4_L * FEATURE4_L];

static float act_conv5[ALEXNET_STATIC_MAX_BATCH * C5_CHANNELS * FEATURE5_L * FEATURE5_L];
static float act_bn5[ALEXNET_STATIC_MAX_BATCH * C5_CHANNELS * FEATURE5_L * FEATURE5_L];
static float act_relu5[ALEXNET_STATIC_MAX_BATCH * C5_CHANNELS * FEATURE5_L * FEATURE5_L];
static float act_mp5[ALEXNET_STATIC_MAX_BATCH * C5_CHANNELS * POOLING5_L * POOLING5_L];

static float act_fc1[ALEXNET_STATIC_MAX_BATCH * FC6_LAYER];
static float act_relu6[ALEXNET_STATIC_MAX_BATCH * FC6_LAYER];
static float act_fc2[ALEXNET_STATIC_MAX_BATCH * FC7_LAYER];
static float act_relu7[ALEXNET_STATIC_MAX_BATCH * FC7_LAYER];
static float act_fc3[ALEXNET_STATIC_MAX_BATCH * OUT_LAYER];

static void zero_f32(float *buf, int n)
{
    for (int i = 0; i < n; i++) {
        buf[i] = 0.0f;
    }
}

void __libc_init_array(void) {}
void __libc_fini_array(void) {}

void metrics(float *ret, int *preds, int *labels, 
                int classes, int totNum, int type)
{
    /**
     * Compute metric on 'preds' and 'labels'
     * 
     * Input:
     *      preds   [totNum]
     *      labels  [totNum]
     *      classes 
     *      totNum
     *      type    
     * Output:
     *      ret     
     * */

    if (classes > OUT_LAYER || classes <= 0) {
        *ret = 0.0f;
        return;
    }

    int *totPred  = metrics_totPred;
    int *totLabel = metrics_totLabel;
    int *TP       = metrics_TP;
    memset(totPred, 0, (size_t)classes * sizeof(int));
    memset(totLabel, 0, (size_t)classes * sizeof(int));
    memset(TP, 0, (size_t)classes * sizeof(int));

    for (int p = 0; p < totNum; p++)
    {
        totPred[preds[p]]++;
        totLabel[labels[p]]++;
        if(preds[p] == labels[p])
        {
            TP[preds[p]]++;
        }
    }

    int tmp_a=0;
    for (int p =0 ; p < classes; p++)
    {
        tmp_a += TP[p];
    }
    float accuracy = tmp_a * 1.0 / totNum;

    if (type == METRIC_ACCURACY)
    {
        *ret = accuracy;
        return;
    }

    float precisions[classes];
    float macro_p = 0;
    for (int p = 0; p < classes; p++)
    {
        precisions[p] = TP[p] / totLabel[p];
        macro_p += precisions[p];
    }
    macro_p /= classes;

    if (type == METRIC_PRECISION)
    {
        *ret = macro_p;
        return;
    }

    float recalls[classes];
    float macro_r = 0;
    for (int p = 0; p < classes; p++)
    {
        recalls[p] = TP[p] / totPred[p];
        macro_r += recalls[p];
    }
    macro_r /= classes;

    if (type == METRIC_RECALL)
    {
        *ret = macro_r;
        return;
    }

    if (type == METRIC_F1SCORE)
    {
        *ret = 2*macro_p*macro_r / (macro_p+macro_r);
        return;
    }
}

int argmax(float *arr, int n)
{
    /**
     * Return the index of max-value among arr ~ arr+n
     * 
     * Input:
     *      arr
     * Output:
     * Return:
     *      the index of max-value
     * */ 
    
    
     /*        for (int i = 0; i < net->batchsize; i++)
                    preds[i] = argmax(net->output + i * net->fc3.out_units, net->fc3.out_units);
    */

    //
    
    int   idx = -1;
    float max = -1111111111;
    for (int p = 0; p<n; p++)
    {
        if (arr[p] > max)
        {
            idx = p;
            max = arr[p];
        }
    }
    assert(idx!=-1);
    return idx;
}


void forward_alexnet(alexnet *net)
{
    if (net->batchsize > ALEXNET_STATIC_MAX_BATCH) {
        printf("Error: batchsize %d exceeds static max batch %d\n", net->batchsize, ALEXNET_STATIC_MAX_BATCH);
        exit(1);
    }

    net->conv1.input = net->input;

    //printf(">>>>>>>>>>>>>>>>>>conv1>>>>>>>>>>>>>>>>>>>>>>>>> \n");
#ifdef SHOW_OP_TIME
#endif
    printf("%d\n", net->batchsize);
    net->conv1.output = act_conv1;
    zero_f32(net->conv1.output, net->batchsize * net->conv1.out_units);
    conv_op_forward(&(net->conv1));
#ifdef SHOW_OP_TIME
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" forward (&(net->conv1)) duration: %.4fs \n", duration);
#endif

#ifdef SHOW_OP_TIME
#endif
    printf("%d\n", net->batchsize);
    net->bn1.output = act_bn1;
    net->bn1.input = net->conv1.output;
    batch_norm_op_forward(&(net->bn1));
#ifdef SHOW_OP_TIME
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" forward (&(net->bn1)) duration: %.4fs \n", duration);
#endif

    net->relu1.output = act_relu1;
    net->relu1.input = net->bn1.output;
    relu_op_forward(&(net->relu1));

    net->mp1.output = act_mp1;
    net->mp1.input = net->relu1.output;
    max_pooling_op_forward(&(net->mp1));

    //printf(">>>>>>>>>>>>>>>>>>conv2>>>>>>>>>>>>>>>>>>>>>>>>> \n");
#ifdef SHOW_OP_TIME
#endif
    net->conv2.output = act_conv2;
    zero_f32(net->conv2.output, net->batchsize * net->conv2.out_units);
    net->conv2.input = net->mp1.output;
    conv_op_forward(&(net->conv2));
#ifdef SHOW_OP_TIME
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" forward (&(net->conv2)) duration: %.4fs \n", duration);
#endif

#ifdef SHOW_OP_TIME
#endif
    net->bn2.output = act_bn2;
    net->bn2.input = net->conv2.output;
    batch_norm_op_forward(&(net->bn2));
#ifdef SHOW_OP_TIME
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" forward (&(net->bn2)) duration: %.4fs \n", duration);
#endif

    net->relu2.output = act_relu2;
    net->relu2.input = net->bn2.output;
    relu_op_forward(&(net->relu2));

    net->mp2.output = act_mp2;
    net->mp2.input = net->relu2.output;
    max_pooling_op_forward(&(net->mp2));

#ifdef SHOW_OP_TIME
#endif
    //printf(">>>>>>>>>>>>>>>>>>conv3>>>>>>>>>>>>>>>>>>>>>>>>> \n");
    net->conv3.output = act_conv3;
    zero_f32(net->conv3.output, net->batchsize * net->conv3.out_units);
    net->conv3.input = net->mp2.output;
    conv_op_forward(&(net->conv3));
#ifdef SHOW_OP_TIME
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" forward (&(net->conv3)) duration: %.4fs \n", duration);
#endif

#ifdef SHOW_OP_TIME
#endif
    net->bn3.output = act_bn3;
    net->bn3.input = net->conv3.output;
    batch_norm_op_forward(&(net->bn3));
#ifdef SHOW_OP_TIME
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" forward (&(net->bn3)); duration: %.4fs \n", duration);
#endif

    net->relu3.output = act_relu3;
    net->relu3.input = net->bn3.output;
    relu_op_forward(&(net->relu3));

    //printf(">>>>>>>>>>>>>>>>>>conv4>>>>>>>>>>>>>>>>>>>>>>>>> \n");
#ifdef SHOW_OP_TIME
#endif
    net->conv4.output = act_conv4;
    zero_f32(net->conv4.output, net->batchsize * net->conv4.out_units);
    net->conv4.input = net->relu3.output;
    conv_op_forward(&(net->conv4));
#ifdef SHOW_OP_TIME
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" forward (&(net->conv4)) duration: %.4fs \n", duration);
#endif

#ifdef SHOW_OP_TIME
#endif
    net->bn4.output = act_bn4;
    net->bn4.input = net->conv4.output;
    batch_norm_op_forward(&(net->bn4));
#ifdef SHOW_OP_TIME
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" forward (&(net->bn4)) duration: %.4fs \n", duration);
#endif

    net->relu4.output = act_relu4;
    net->relu4.input = net->bn4.output;
    relu_op_forward(&(net->relu4));

    //printf(">>>>>>>>>>>>>>>>>>conv5>>>>>>>>>>>>>>>>>>>>>>>>> \n");
#ifdef SHOW_OP_TIME
#endif
    net->conv5.output = act_conv5;
    zero_f32(net->conv5.output, net->batchsize * net->conv5.out_units);
    net->conv5.input = net->relu4.output;
    conv_op_forward(&(net->conv5));
#ifdef SHOW_OP_TIME
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" forward (&(net->conv5)) duration: %.4fs \n", duration);
#endif

#ifdef SHOW_OP_TIME
#endif
    net->bn5.output = act_bn5;
    net->bn5.input = net->conv5.output;
    batch_norm_op_forward(&(net->bn5));
#ifdef SHOW_OP_TIME
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" forward (&(net->bn5)) duration: %.4fs \n", duration);
#endif

    net->relu5.output = act_relu5;
    net->relu5.input = net->bn5.output;
    relu_op_forward(&(net->relu5));

    net->mp5.output = act_mp5;
    net->mp5.input = net->relu5.output;
    max_pooling_op_forward(&(net->mp5));


    dropout(net->mp5.output, DROPOUT_PROB, net->mp5.batchsize * net->mp5.out_units);

#ifdef SHOW_OP_TIME
#endif
    net->fc1.output = act_fc1;
    zero_f32(net->fc1.output, net->batchsize * net->fc1.out_units);
    net->fc1.input = net->mp5.output;
    fc_op_forward(&(net->fc1));
#ifdef SHOW_OP_TIME
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" forward (&(net->fc1)) duration: %.4fs \n", duration);
#endif

    net->relu6.output = act_relu6;
    net->relu6.input = net->fc1.output;
    relu_op_forward(&(net->relu6));

    dropout(net->relu6.output, DROPOUT_PROB, net->relu6.batchsize * net->relu6.units);

#ifdef SHOW_OP_TIME
#endif
    net->fc2.output = act_fc2;
    zero_f32(net->fc2.output, net->batchsize * net->fc2.out_units);
    net->fc2.input = net->relu6.output;
    fc_op_forward(&(net->fc2));
#ifdef SHOW_OP_TIME
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" forward (&(net->fc2)) duration: %.4fs \n", duration);
#endif

    for(int p=0; p< net->fc2.out_units * net->batchsize; p++)
    {
        if(net->fc2.output[p]<(0-64) | net->fc2.output[p]>64)
        {
            printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!Forward: fc2 too big/small !!! %f \n", net->fc2.output[p]);
            break;
        }
    }

    net->relu7.output = act_relu7;
    net->relu7.input = net->fc2.output;
    relu_op_forward(&(net->relu7));

#ifdef SHOW_OP_TIME
#endif
    net->fc3.output = act_fc3;
    zero_f32(net->fc3.output, net->batchsize * net->fc3.out_units);
    net->fc3.input = net->relu7.output;
    fc_op_forward(&(net->fc3));
#ifdef SHOW_OP_TIME
    duration = (finish.tv_sec - start.tv_sec);
    duration += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf(" forward (&(net->fc3) duration: %.4fs \n", duration);
#endif

    net->output = net->fc3.output;
}


void free_forward_activations(alexnet *net)
{
    /**
     * Free all heap buffers allocated by forward_alexnet.
     * Must be called after each forward pass when NOT followed by backward_alexnet
     * (which already frees these buffers itself).
     */
    net->conv1.output = NULL;
    net->bn1.output = NULL;
    net->relu1.output = NULL;
    net->mp1.output = NULL;

    net->conv2.output = NULL;
    net->bn2.output = NULL;
    net->relu2.output = NULL;
    net->mp2.output = NULL;

    net->conv3.output = NULL;
    net->bn3.output = NULL;
    net->relu3.output = NULL;

    net->conv4.output = NULL;
    net->bn4.output = NULL;
    net->relu4.output = NULL;

    net->conv5.output = NULL;
    net->bn5.output = NULL;
    net->relu5.output = NULL;
    net->mp5.output = NULL;

    net->fc1.output = NULL;
    net->relu6.output = NULL;
    net->fc2.output = NULL;
    net->relu7.output = NULL;
    net->fc3.output = NULL;
    net->output = NULL;

    /* batchnorm internals are static in batchnorm_layer.c */
    net->bn1.x_norm = NULL;
    net->bn2.x_norm = NULL;
    net->bn3.x_norm = NULL;
    net->bn4.x_norm = NULL;
    net->bn5.x_norm = NULL;
    net->bn1.var = NULL;
    net->bn2.var = NULL;
    net->bn3.var = NULL;
    net->bn4.var = NULL;
    net->bn5.var = NULL;
    net->bn1.avg = NULL;
    net->bn2.avg = NULL;
    net->bn3.avg = NULL;
    net->bn4.avg = NULL;
    net->bn5.avg = NULL;
}


void malloc_alexnet(alexnet *net)
{
    net->conv1.weights = conv1_weights;
    net->conv2.weights = conv2_weights;
    net->conv3.weights = conv3_weights;
    net->conv4.weights = conv4_weights;
    net->conv5.weights = conv5_weights;
    net->fc1.weights = fc1_weights;
    net->fc2.weights = fc2_weights;
    net->fc3.weights = fc3_weights;

    net->conv1.bias = conv1_bias;
    net->conv2.bias = conv2_bias;
    net->conv3.bias = conv3_bias;
    net->conv4.bias = conv4_bias;
    net->conv5.bias = conv5_bias;
    net->fc1.bias = fc1_bias;
    net->fc2.bias = fc2_bias;
    net->fc3.bias = fc3_bias;

    net->bn1.gamma = bn1_gamma;
    net->bn2.gamma = bn2_gamma;
    net->bn3.gamma = bn3_gamma;
    net->bn4.gamma = bn4_gamma;
    net->bn5.gamma = bn5_gamma;
    net->bn1.beta = bn1_beta;
    net->bn2.beta = bn2_beta;
    net->bn3.beta = bn3_beta;
    net->bn4.beta = bn4_beta;
    net->bn5.beta = bn5_beta;
}

void free_alexnet(alexnet *net)
{
    net->conv1.weights = NULL;
    net->conv2.weights = NULL;
    net->conv3.weights = NULL;
    net->conv4.weights = NULL;
    net->conv5.weights = NULL;
    net->fc1.weights = NULL;
    net->fc2.weights = NULL;
    net->fc3.weights = NULL;

    net->conv1.bias = NULL;
    net->conv2.bias = NULL;
    net->conv3.bias = NULL;
    net->conv4.bias = NULL;
    net->conv5.bias = NULL;
    net->fc1.bias = NULL;
    net->fc2.bias = NULL;
    net->fc3.bias = NULL;

    net->bn1.gamma = NULL;
    net->bn2.gamma = NULL;
    net->bn3.gamma = NULL;
    net->bn4.gamma = NULL;
    net->bn5.gamma = NULL;
    net->bn1.beta = NULL;
    net->bn2.beta = NULL;
    net->bn3.beta = NULL;
    net->bn4.beta = NULL;
    net->bn5.beta = NULL;
}

void bind_alexnet_static_memory(alexnet *net)
{
    malloc_alexnet(net);
}

void release_alexnet_static_memory(alexnet *net)
{
    free_alexnet(net);
}

static void gauss_initialization(float *p, int n, int in_units, int out_units)
{
    float mean  = 0;
    float stddv = 0.01;

	float V1, V2, S, X;
	static int phase = 0;
    for (int shift = 0; shift < n; shift++)
    {
        if (phase == 0) {
            do {
                float U1 = (float) rand() / RAND_MAX;
                float U2 = (float) rand() / RAND_MAX;

                V1 = 2 * U1 - 1;
                V2 = 2 * U2 - 1;
                S = V1 * V1 + V2 * V2;
            } while (S >= 1 || S == 0);
    
            X = V1 * sqrt(-2 * log(S) / S);
        }else {
            X = V2 * sqrt(-2 * log(S) / S);
        }
        phase = 1 - phase;

        p[shift] = mean + stddv * X;
    }
}


//
// save trainable weights of network
//
void save_alexnet(alexnet *net)
{
    /**
     * save weights of net to file
     * */
    // FILE *fp = fopen(filename, "wb");
    // save_conv_weights(&(net->conv1), fp);
    // save_conv_weights(&(net->conv2), fp);
    // save_conv_weights(&(net->conv3), fp);
    // save_conv_weights(&(net->conv4), fp);
    // save_conv_weights(&(net->conv5), fp);
    // save_fc_weights(&(net->fc1), fp);
    // save_fc_weights(&(net->fc2), fp);
    // save_fc_weights(&(net->fc3), fp);
    // save_batchnorm_weights(&(net->bn1), fp);
    // save_batchnorm_weights(&(net->bn2), fp);
    // save_batchnorm_weights(&(net->bn3), fp);
    // save_batchnorm_weights(&(net->bn4), fp);
    // save_batchnorm_weights(&(net->bn5), fp);
    // fclose(fp);
    printf("NOT SAVED WEIGHTS\n");
}

// static uint64_t checksum_f32(const float *arr, size_t n)
// {
//     // FNV-1a on raw float bytes for deterministic run-to-run comparison.
//     const unsigned char *bytes = (const unsigned char *)arr;
//     size_t len = n * sizeof(float);
//     uint64_t hash = 1469598103934665603ULL;
//     for (size_t i = 0; i < len; i++) {
//         hash ^= (uint64_t)bytes[i];
//         hash *= 1099511628211ULL;
//     }
//     return hash;
// }

static int verify_weight_array_shapes(void)
{
    int ok = 1;

    #define CHECK_SHAPE(name, expected) \
        do { \
            size_t got = sizeof(name) / sizeof((name)[0]); \
            size_t exp = (size_t)(expected); \
            if (got != exp) { \
                printf("[shape-error] %s: got=%zu expected=%zu\\n", #name, got, exp); \
                ok = 0; \
            } \
        } while (0)

    CHECK_SHAPE(conv1_weights, C1_CHANNELS * IN_CHANNELS * C1_KERNEL_L * C1_KERNEL_L);
    CHECK_SHAPE(conv1_bias, C1_CHANNELS);
    CHECK_SHAPE(conv2_weights, C2_CHANNELS * C1_CHANNELS * C2_KERNEL_L * C2_KERNEL_L);
    CHECK_SHAPE(conv2_bias, C2_CHANNELS);
    CHECK_SHAPE(conv3_weights, C3_CHANNELS * C2_CHANNELS * C3_KERNEL_L * C3_KERNEL_L);
    CHECK_SHAPE(conv3_bias, C3_CHANNELS);
    CHECK_SHAPE(conv4_weights, C4_CHANNELS * C3_CHANNELS * C4_KERNEL_L * C4_KERNEL_L);
    CHECK_SHAPE(conv4_bias, C4_CHANNELS);
    CHECK_SHAPE(conv5_weights, C5_CHANNELS * C4_CHANNELS * C5_KERNEL_L * C5_KERNEL_L);
    CHECK_SHAPE(conv5_bias, C5_CHANNELS);

    CHECK_SHAPE(fc1_weights, C5_CHANNELS * FC6_LAYER * POOLING5_L * POOLING5_L);
    CHECK_SHAPE(fc1_bias, FC6_LAYER);
    CHECK_SHAPE(fc2_weights, FC6_LAYER * FC7_LAYER);
    CHECK_SHAPE(fc2_bias, FC7_LAYER);
    CHECK_SHAPE(fc3_weights, FC7_LAYER * OUT_LAYER);
    CHECK_SHAPE(fc3_bias, OUT_LAYER);

    CHECK_SHAPE(bn1_gamma, C1_CHANNELS);
    CHECK_SHAPE(bn1_beta, C1_CHANNELS);
    CHECK_SHAPE(bn2_gamma, C2_CHANNELS);
    CHECK_SHAPE(bn2_beta, C2_CHANNELS);
    CHECK_SHAPE(bn3_gamma, C3_CHANNELS);
    CHECK_SHAPE(bn3_beta, C3_CHANNELS);
    CHECK_SHAPE(bn4_gamma, C4_CHANNELS);
    CHECK_SHAPE(bn4_beta, C4_CHANNELS);
    CHECK_SHAPE(bn5_gamma, C5_CHANNELS);
    CHECK_SHAPE(bn5_beta, C5_CHANNELS);

    #undef CHECK_SHAPE
    return ok;
}

// static void print_weight_checksums(void)
// {
//     printf("Weight checksums (FNV1a64):\n");
//     printf("  conv1_weights: 0x%016llx\n", (unsigned long long)checksum_f32(conv1_weights, sizeof(conv1_weights)/sizeof(conv1_weights[0])));
//     printf("  conv4_weights: 0x%016llx\n", (unsigned long long)checksum_f32(conv4_weights, sizeof(conv4_weights)/sizeof(conv4_weights[0])));
//     printf("  conv5_weights: 0x%016llx\n", (unsigned long long)checksum_f32(conv5_weights, sizeof(conv5_weights)/sizeof(conv5_weights[0])));
//     printf("  fc2_weights:   0x%016llx\n", (unsigned long long)checksum_f32(fc2_weights, sizeof(fc2_weights)/sizeof(fc2_weights[0])));
//     printf("  fc3_weights:   0x%016llx\n", (unsigned long long)checksum_f32(fc3_weights, sizeof(fc3_weights)/sizeof(fc3_weights[0])));
//     printf("  bn4_gamma:     0x%016llx\n", (unsigned long long)checksum_f32(bn4_gamma, sizeof(bn4_gamma)/sizeof(bn4_gamma[0])));
//     printf("  bn5_gamma:     0x%016llx\n", (unsigned long long)checksum_f32(bn5_gamma, sizeof(bn5_gamma)/sizeof(bn5_gamma[0])));
// }

//
// load trainable weights of network
//
// void load_alexnet(alexnet *net)
// {
//     /**
//      * load weights of network from file
//      * */
//     FILE *fp = fopen(filename, "rb");
//     if (fp == NULL) {
//         fprintf(stderr, "Error: Cannot open weights file \"%s\"\n", filename);
//         exit(1);
//     }
//     load_conv_weights(&(net->conv1), fp);
//     load_conv_weights(&(net->conv2), fp);
//     load_conv_weights(&(net->conv3), fp);
//     load_conv_weights(&(net->conv4), fp);
//     load_conv_weights(&(net->conv5), fp);
//     load_fc_weights(&(net->fc1), fp);
//     load_fc_weights(&(net->fc2), fp);
//     load_fc_weights(&(net->fc3), fp);
//     load_batchnorm_weights(&(net->bn1), fp);
//     load_batchnorm_weights(&(net->bn2), fp);
//     load_batchnorm_weights(&(net->bn3), fp);
//     load_batchnorm_weights(&(net->bn4), fp);
//     load_batchnorm_weights(&(net->bn5), fp);
//     fclose(fp);
//     printf("Load weights from \"%s\" successfully... \n", filename);
// }

void load_alexnet(alexnet *net)
{
    /**
     * Validate shape compatibility for compile-time arrays.
     */
    (void)net;
     
    if (!verify_weight_array_shapes()) {
        printf("Fatal: weight array shape mismatch detected.\n");
        exit(1);
    }

    // print_weight_checksums();

    printf("Network pointers use weights.c arrays directly (no parameter buffer copy).\n");
}

void alexnet_set_all_trainable(alexnet *net, short trainable)
{
    net->trainable.conv1 = trainable;
    net->trainable.conv2 = trainable;
    net->trainable.conv3 = trainable;
    net->trainable.conv4 = trainable;
    net->trainable.conv5 = trainable;

    net->trainable.bn1 = trainable;
    net->trainable.bn2 = trainable;
    net->trainable.bn3 = trainable;
    net->trainable.bn4 = trainable;
    net->trainable.bn5 = trainable;

    net->trainable.fc1 = trainable;
    net->trainable.fc2 = trainable;
    net->trainable.fc3 = trainable;
}

static void print_trainable_layers(const alexnet *net)
{
    int enabled = 0;

    printf("Trainable layers:\n");
    printf("  conv1: %d\n", net->trainable.conv1); enabled += net->trainable.conv1 ? 1 : 0;
    printf("  conv2: %d\n", net->trainable.conv2); enabled += net->trainable.conv2 ? 1 : 0;
    printf("  conv3: %d\n", net->trainable.conv3); enabled += net->trainable.conv3 ? 1 : 0;
    printf("  conv4: %d\n", net->trainable.conv4); enabled += net->trainable.conv4 ? 1 : 0;
    printf("  conv5: %d\n", net->trainable.conv5); enabled += net->trainable.conv5 ? 1 : 0;

    printf("  bn1: %d\n", net->trainable.bn1); enabled += net->trainable.bn1 ? 1 : 0;
    printf("  bn2: %d\n", net->trainable.bn2); enabled += net->trainable.bn2 ? 1 : 0;
    printf("  bn3: %d\n", net->trainable.bn3); enabled += net->trainable.bn3 ? 1 : 0;
    printf("  bn4: %d\n", net->trainable.bn4); enabled += net->trainable.bn4 ? 1 : 0;
    printf("  bn5: %d\n", net->trainable.bn5); enabled += net->trainable.bn5 ? 1 : 0;

    printf("  fc1: %d\n", net->trainable.fc1); enabled += net->trainable.fc1 ? 1 : 0;
    printf("  fc2: %d\n", net->trainable.fc2); enabled += net->trainable.fc2 ? 1 : 0;
    printf("  fc3: %d\n", net->trainable.fc3); enabled += net->trainable.fc3 ? 1 : 0;

    printf("Total trainable layers: %d/13\n", enabled);
}

void setup_alexnet(alexnet *net, short batchsize)
{
    /**
     * initialize alexnet
     * */
    net->batchsize = batchsize;
    printf("batchsize in setup\n");
    net->conv1.batchsize = batchsize;
    net->conv2.batchsize = batchsize;
    net->conv3.batchsize = batchsize;
    net->conv4.batchsize = batchsize;
    net->conv5.batchsize = batchsize;
    net->fc1.batchsize   = batchsize;
    net->fc2.batchsize   = batchsize;
    net->fc3.batchsize   = batchsize;
    net->bn1.batchsize   = batchsize;
    net->bn2.batchsize   = batchsize;
    net->bn3.batchsize   = batchsize;
    net->bn4.batchsize   = batchsize;
    net->bn5.batchsize   = batchsize;
    net->mp1.batchsize   = batchsize;
    net->mp2.batchsize   = batchsize;
    net->mp5.batchsize   = batchsize;
    net->relu1.batchsize = batchsize;
    net->relu2.batchsize = batchsize;
    net->relu3.batchsize = batchsize;
    net->relu4.batchsize = batchsize;
    net->relu5.batchsize = batchsize;
    net->relu6.batchsize = batchsize;
    net->relu7.batchsize = batchsize;


    net->conv1.in_channels = IN_CHANNELS;
    net->conv1.out_channels = C1_CHANNELS;
    net->conv1.in_h = FEATURE0_L;
    net->conv1.in_w = FEATURE0_L;
    net->conv1.kernel_size = C1_KERNEL_L;
    net->conv1.padding = C1_PADDING;
    net->conv1.stride = C1_STRIDES;
    net->conv1.out_h = FEATURE1_L;
    net->conv1.out_w = FEATURE1_L;
    net->conv1.in_units = IN_CHANNELS*FEATURE0_L*FEATURE0_L;
    net->conv1.out_units = C1_CHANNELS*FEATURE1_L*FEATURE1_L;
    net->conv1.layer_id = 1;

    net->bn1.units = net->conv1.out_units;
    net->bn1.channels = C1_CHANNELS;
    net->bn1.spatial_size = FEATURE1_L * FEATURE1_L;
    net->bn1.layer_id = 1;

    net->relu1.units = net->bn1.units;
    
    net->mp1.channels = C1_CHANNELS;
    net->mp1.stride = 2; //max pooling stride
    net->mp1.kernel_size = 2;
    // net->mp1.kernel_size = 3;

    net->mp1.in_h = FEATURE1_L;
    net->mp1.in_w = FEATURE1_L;
    net->mp1.out_w = POOLING1_L;
    net->mp1.out_h = POOLING1_L;
    net->mp1.in_units = net->relu1.units;
    net->mp1.out_units = C1_CHANNELS*POOLING1_L*POOLING1_L;

    net->conv2.in_channels = C1_CHANNELS;
    net->conv2.out_channels = C2_CHANNELS;
    net->conv2.in_h = POOLING1_L;
    net->conv2.in_w = POOLING1_L;
    net->conv2.kernel_size = C2_KERNEL_L;
    net->conv2.padding = C2_PADDING;
    net->conv2.stride = C2_STRIDES;
    net->conv2.out_h = FEATURE2_L;
    net->conv2.out_w = FEATURE2_L;
    net->conv2.in_units = net->mp1.out_units;
    net->conv2.out_units = C2_CHANNELS*FEATURE2_L*FEATURE2_L;
    net->conv2.layer_id = 2;

    net->bn2.units = net->conv2.out_units;
    net->bn2.channels = C2_CHANNELS;
    net->bn2.spatial_size = FEATURE2_L * FEATURE2_L;
    net->bn2.layer_id = 2;

    net->relu2.units = net->bn2.units;

    net->mp2.channels = C2_CHANNELS;
    net->mp2.stride = 2;
    net->mp2.kernel_size = 2;
    net->mp2.in_h = FEATURE2_L;
    net->mp2.in_w = FEATURE2_L;
    net->mp2.out_w = POOLING2_L;
    net->mp2.out_h = POOLING2_L;
    net->mp2.in_units = net->relu2.units;
    net->mp2.out_units = C2_CHANNELS*POOLING2_L*POOLING2_L;

    net->conv3.in_channels = C2_CHANNELS;
    net->conv3.out_channels = C3_CHANNELS;
    net->conv3.in_h = POOLING2_L;
    net->conv3.in_w = POOLING2_L;
    net->conv3.kernel_size = C3_KERNEL_L;
    net->conv3.padding = C3_PADDING;
    net->conv3.stride = C3_STRIDES;
    net->conv3.out_h = FEATURE3_L;
    net->conv3.out_w = FEATURE3_L;
    net->conv3.in_units = net->mp2.out_units;
    net->conv3.out_units = C3_CHANNELS*FEATURE3_L*FEATURE3_L;
    net->conv3.layer_id = 3;

    net->bn3.units = net->conv3.out_units;
    net->bn3.channels = C3_CHANNELS;
    net->bn3.spatial_size = FEATURE3_L * FEATURE3_L;
    net->bn3.layer_id = 3;

    net->relu3.units = net->bn3.units;

    net->conv4.in_channels = C3_CHANNELS;
    net->conv4.out_channels = C4_CHANNELS;
    net->conv4.in_h = FEATURE3_L;
    net->conv4.in_w = FEATURE3_L;
    net->conv4.kernel_size = C4_KERNEL_L;
    net->conv4.padding = C4_PADDING;
    net->conv4.stride = C4_STRIDES;
    net->conv4.out_h = FEATURE4_L;
    net->conv4.out_w = FEATURE4_L;
    net->conv4.in_units = net->relu3.units;
    net->conv4.out_units = C4_CHANNELS*FEATURE4_L*FEATURE4_L;
    net->conv4.layer_id = 4;

    net->bn4.units = net->conv4.out_units;
    net->bn4.channels = C4_CHANNELS;
    net->bn4.spatial_size = FEATURE4_L * FEATURE4_L;
    net->bn4.layer_id = 4;

    net->relu4.units = net->bn4.units;

    net->conv5.in_channels = C4_CHANNELS;
    net->conv5.out_channels = C5_CHANNELS;
    net->conv5.in_h = FEATURE5_L;
    net->conv5.in_w = FEATURE5_L;
    net->conv5.kernel_size = C5_KERNEL_L;
    net->conv5.padding = C5_PADDING;
    net->conv5.stride = C5_STRIDES;
    net->conv5.out_h = FEATURE5_L;
    net->conv5.out_w = FEATURE5_L;
    net->conv5.in_units = net->relu4.units;
    net->conv5.out_units = C5_CHANNELS*FEATURE5_L*FEATURE5_L;
    net->conv5.layer_id = 5;

    net->bn5.units = net->conv5.out_units;
    net->bn5.channels = C5_CHANNELS;
    net->bn5.spatial_size = FEATURE5_L * FEATURE5_L;
    net->bn5.layer_id = 5;

    net->relu5.units = net->bn5.units;

    net->mp5.channels = C5_CHANNELS;
    net->mp5.stride = 2;
    net->mp5.kernel_size = 2;
    net->mp5.in_h = FEATURE5_L;
    net->mp5.in_w = FEATURE5_L;
    net->mp5.out_w = POOLING5_L;
    net->mp5.out_h = POOLING5_L;
    net->mp5.in_units = net->relu5.units;
    net->mp5.out_units = C5_CHANNELS*POOLING5_L*POOLING5_L;

    net->fc1.in_units = net->mp5.out_units;
    net->fc1.out_units = FC6_LAYER;
    net->fc1.layer_id = 1;
    
    net->relu6.units = FC6_LAYER; 
    
    net->fc2.in_units = FC6_LAYER;
    net->fc2.out_units = FC7_LAYER;
    net->fc2.layer_id = 2;

    net->relu7.units = FC7_LAYER;

    net->fc3.in_units = FC7_LAYER;
    net->fc3.out_units = OUT_LAYER;
    net->fc3.layer_id = 3;

    alexnet_set_all_trainable(net, 1);
}

void alexnet_init_weights(alexnet *net)
{
    if (net == NULL) {
        return;
    }

    /*
     * In training builds, embedded arrays may be placeholders; initialize
     * weights randomly unless embedded loading is explicitly enabled.
     */
#if defined(ALEXNET_MODE_TRAIN) || defined(INFERENCE_MODE) || defined(ALEXNET_USE_EMBEDDED_WEIGHTS)
    load_alexnet(net);
    return;
#endif

    // initialize weights for this network
    gauss_initialization(net->conv1.weights, C1_CHANNELS*IN_CHANNELS*C1_KERNEL_L*C1_KERNEL_L, net->conv1.in_units, net->conv1.out_units);
    gauss_initialization(net->conv2.weights, C2_CHANNELS*C1_CHANNELS*C2_KERNEL_L*C2_KERNEL_L, net->conv2.in_units, net->conv2.out_units);
    gauss_initialization(net->conv3.weights, C3_CHANNELS*C2_CHANNELS*C3_KERNEL_L*C3_KERNEL_L, net->conv3.in_units, net->conv3.out_units);
    gauss_initialization(net->conv4.weights, C4_CHANNELS*C3_CHANNELS*C4_KERNEL_L*C4_KERNEL_L, net->conv4.in_units, net->conv4.out_units);
    gauss_initialization(net->conv5.weights, C5_CHANNELS*C4_CHANNELS*C5_KERNEL_L*C5_KERNEL_L, net->conv5.in_units, net->conv5.out_units);
    gauss_initialization(net->fc1.weights, C5_CHANNELS*FC6_LAYER*POOLING5_L*POOLING5_L, net->fc1.in_units, net->fc1.out_units);
    gauss_initialization(net->fc2.weights, FC6_LAYER*FC7_LAYER, net->fc2.in_units, net->fc2.out_units);
    gauss_initialization(net->fc3.weights, FC7_LAYER*OUT_LAYER, net->fc3.in_units, net->fc3.out_units);

    int i;
    for(i=0; i<C1_CHANNELS; i++)
        net->conv1.bias[i] = 1;
    for(i=0; i<C2_CHANNELS; i++)
        net->conv2.bias[i] = 1;
    for(i=0; i<C3_CHANNELS; i++)
        net->conv3.bias[i] = 1;
    for(i=0; i<C4_CHANNELS; i++)
        net->conv4.bias[i] = 1;
    for(i=0; i<C5_CHANNELS; i++)
        net->conv5.bias[i] = 1;
    for(i=0; i<FC6_LAYER; i++)
        net->fc1.bias[i] = 1;
    for(i=0; i<FC7_LAYER; i++)
        net->fc2.bias[i] = 1;
    for(i=0; i<OUT_LAYER; i++)
        net->fc3.bias[i] = 1;

    for(i=0; i<(net->bn1.channels); i++)
        net->bn1.gamma[i] = 1;
    for(i=0; i<(net->bn2.channels); i++)
        net->bn2.gamma[i] = 1;
    for(i=0; i<(net->bn3.channels); i++)
        net->bn3.gamma[i] = 1;
    for(i=0; i<(net->bn4.channels); i++)
        net->bn4.gamma[i] = 1;
    for(i=0; i<(net->bn5.channels); i++)
        net->bn5.gamma[i] = 1;
    
    memset(net->bn1.beta, 0, sizeof(float)*(net->bn1.channels));
    memset(net->bn2.beta, 0, sizeof(float)*(net->bn2.channels));
    memset(net->bn3.beta, 0, sizeof(float)*(net->bn3.channels));
    memset(net->bn4.beta, 0, sizeof(float)*(net->bn4.channels));
    memset(net->bn5.beta, 0, sizeof(float)*(net->bn5.channels));
}


#ifndef ALEXNET_BATCHSIZE
#define ALEXNET_BATCHSIZE 1
#endif

#ifndef ALEXNET_EPOCHS
#define ALEXNET_EPOCHS 1000
#endif

#ifndef ALEXNET_INFER_IDX
#define ALEXNET_INFER_IDX -1
#endif

int main(void)
{
    /**
     * 
     * Entrance
     * 
     * */
    static alexnet net;
    
    #if defined(ALEXNET_MODE_TRAIN)
    printf("batch size: %d \n", ALEXNET_BATCHSIZE);
    printf("epochs: %d \n", ALEXNET_EPOCHS);
    printf("net: %p\n", &net);
    setup_alexnet(&net, ALEXNET_BATCHSIZE);
    printf("setup finished\n");
    bind_alexnet_static_memory(&net);
    printf("allocation of net struct\n");
    alexnet_init_weights(&net);

    /* Train only selected layers: conv4, conv5, fc2, fc3 */
    alexnet_set_all_trainable(&net, 0);
    // net.trainable.conv4 = 1;
    net.trainable.conv5 = 1;
    // net.trainable.fc2 = 1;
    net.trainable.fc3 = 1;
    print_trainable_layers(&net);

    alexnet_train(&net, ALEXNET_EPOCHS);
    release_alexnet_static_memory(&net);
    #elif defined(ALEXNET_MODE_INFERENCE)
    const unsigned char *infer_bytes = img_data;
    if (ALEXNET_INFER_IDX >= 0) {
        if (ALEXNET_INFER_IDX >= cifar10_count) {
            printf("Error: ALEXNET_INFER_IDX %d out of range [0, %d)\n", ALEXNET_INFER_IDX, cifar10_count);
            return 1;
        }
        infer_bytes = cifar10_data + cifar10_offsets[ALEXNET_INFER_IDX];
        printf("inference sample: CIFAR-10 idx=%d label=%d\n", ALEXNET_INFER_IDX, cifar10_labels[ALEXNET_INFER_IDX]);
    }
    setup_alexnet(&net, 1);
    bind_alexnet_static_memory(&net);
    alexnet_init_weights(&net);
    printf("alexnet setup fininshed. Waiting for inference...\n");
    alexnet_inference(&net, infer_bytes);
    release_alexnet_static_memory(&net);
    #else
    printf("Error: define ALEXNET_MODE_TRAIN or ALEXNET_MODE_INFERENCE at compile time.\n");
    return 1;
    #endif

    return 0;
}
