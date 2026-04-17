//
// File:        alexnet.h
// Description: alexnet.h
// Author:      Haris Wang
//
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#ifdef SPIKE
#include <printf.h>
#elif defined ARA_LINUX
#include <stdio.h>
#else
#include "printf.h"
#endif
#include "convolution_layer.h"
#include "maxpooling_layer.h"
#include "activation_layer.h"
#include "fc_layer.h"
#include "batchnorm_layer.h"
#include "dropout_layer.h"

#define SHOW_PREDCITION_DETAIL
//#define SHOW_METRIC_EVALUTE
//#define SHOW_OP_TIME


//
//  Definition of model shape
//

//so that amount of kernels catching characteristics of the image
#define IN_CHANNELS 3
#define C1_CHANNELS 64
#define C2_CHANNELS 128
#define C3_CHANNELS 256
#define C4_CHANNELS 256
#define C5_CHANNELS 128

#define C1_KERNEL_L 3    // instead of 11
#define C2_KERNEL_L 3    // instead of 5
#define C3_KERNEL_L 3
#define C4_KERNEL_L 3
#define C5_KERNEL_L 3   // pointless to be just 1...

#define C1_STRIDES 1
#define C2_STRIDES 1
#define C3_STRIDES 1
#define C4_STRIDES 1
#define C5_STRIDES 1

#define C1_PADDING 1
#define C2_PADDING 1
#define C3_PADDING 1
#define C4_PADDING 1
#define C5_PADDING 1


#define FEATURE0_L 32
#define FEATURE1_L 32
#define POOLING1_L 16
#define FEATURE2_L 16
#define POOLING2_L 8
#define FEATURE3_L 8
#define FEATURE4_L 8
#define FEATURE5_L 8
#define POOLING5_L 4

#define FC6_LAYER   512
#define FC7_LAYER   512
// #define OUT_LAYER   1000 FOR IMAGENET
#define OUT_LAYER   10 // FOR TINY IMAGENET

#define DROPOUT_PROB  0.0


typedef struct network {

    float *input;
    float *output;
    short batchsize;
    
    conv_op conv1;
    batch_norm_op bn1;
    nonlinear_op relu1;

    max_pooling_op mp1;

    conv_op conv2;
    batch_norm_op bn2;
    nonlinear_op relu2;

    max_pooling_op mp2;

    conv_op conv3;
    batch_norm_op bn3;
    nonlinear_op relu3;

    conv_op conv4;
    batch_norm_op bn4;
    nonlinear_op relu4;

    conv_op conv5;
    batch_norm_op bn5;
    nonlinear_op relu5;

    max_pooling_op mp5;

    fc_op fc1;
    nonlinear_op relu6;

    fc_op fc2;
    nonlinear_op relu7;

    fc_op fc3;

    struct {
        short conv1, conv2, conv3, conv4, conv5;
        short bn1, bn2, bn3, bn4, bn5;
        short fc1, fc2, fc3;
    } trainable;
} alexnet;


//
//  Definiation of metric type
//
#define METRIC_ACCURACY  0
#define METRIC_PRECISION 1      // macro-precision
#define METRIC_RECALL    2      // macro-recall
#define METRIC_F1SCORE   3
#define METRIC_ROC       4

void metrics(float *ret, int *preds, int *labels, 
                int classes, int TotalNum, int type);
int argmax(float *arr, int n);


void malloc_alexnet(alexnet *net);
void free_alexnet(alexnet *net);
void bind_alexnet_static_memory(alexnet *net);
void release_alexnet_static_memory(alexnet *net);
void alexnet_set_all_trainable(alexnet *net, short trainable);

void set_alexnet(alexnet *net, short batchsize, char *weights_path);

void forward_alexnet(alexnet *net);
void free_forward_activations(alexnet *net);
void backward_alexnet(alexnet *net, int *batch_Y);

void alexnet_train(alexnet *net, int epochs);
void alexnet_test(alexnet *net);
void alexnet_inference(alexnet *net, const unsigned char *img_bytes);
void compute_batch_metrics(const int *preds, const int *labels, int batchsize);
