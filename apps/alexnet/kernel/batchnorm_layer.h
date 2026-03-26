//
// File:        batchnorm_layer.h
// Description: interface of batch normalization layer
// Author:      Haris Wang
//
#include <stdlib.h>
#include <math.h>

#define EPSILON 0.00001

#ifndef ALEXNET_STATIC_MAX_BATCH
#define ALEXNET_STATIC_MAX_BATCH 400
#endif

#define BN1_CHANNELS 64
#define BN2_CHANNELS 128
#define BN3_CHANNELS 256
#define BN4_CHANNELS 256
#define BN5_CHANNELS 128

#define BN1_SPATIAL (32 * 32)
#define BN2_SPATIAL (16 * 16)
#define BN3_SPATIAL (8 * 8)
#define BN4_SPATIAL (8 * 8)
#define BN5_SPATIAL (8 * 8)

#define BN1_UNITS (BN1_CHANNELS * BN1_SPATIAL)
#define BN2_UNITS (BN2_CHANNELS * BN2_SPATIAL)
#define BN3_UNITS (BN3_CHANNELS * BN3_SPATIAL)
#define BN4_UNITS (BN4_CHANNELS * BN4_SPATIAL)
#define BN5_UNITS (BN5_CHANNELS * BN5_SPATIAL)

typedef struct batch_norm_op {
    float *input; float *d_input;
    float *output; float *d_output;
    float *gamma; float *d_gamma;
    float *beta; float *d_beta;

    int units;
    int channels;
    int spatial_size;
    short batchsize;
    short layer_id;
    
    float *x_norm;
    float *avg;
    float *var;
} batch_norm_op;

extern float bn1_avg_buf[BN1_CHANNELS];
extern float bn2_avg_buf[BN2_CHANNELS];
extern float bn3_avg_buf[BN3_CHANNELS];
extern float bn4_avg_buf[BN4_CHANNELS];
extern float bn5_avg_buf[BN5_CHANNELS];

extern float bn1_var_buf[BN1_CHANNELS];
extern float bn2_var_buf[BN2_CHANNELS];
extern float bn3_var_buf[BN3_CHANNELS];
extern float bn4_var_buf[BN4_CHANNELS];
extern float bn5_var_buf[BN5_CHANNELS];

extern float bn1_x_norm_buf[ALEXNET_STATIC_MAX_BATCH * BN1_UNITS];
extern float bn2_x_norm_buf[ALEXNET_STATIC_MAX_BATCH * BN2_UNITS];
extern float bn3_x_norm_buf[ALEXNET_STATIC_MAX_BATCH * BN3_UNITS];
extern float bn4_x_norm_buf[ALEXNET_STATIC_MAX_BATCH * BN4_UNITS];
extern float bn5_x_norm_buf[ALEXNET_STATIC_MAX_BATCH * BN5_UNITS];

extern float bn_dxnorm_scratch[ALEXNET_STATIC_MAX_BATCH * BN3_UNITS];

void batch_norm_op_forward(batch_norm_op *op);
void batch_norm_op_backward(batch_norm_op *op);
void batch_norm_op_backward_full(batch_norm_op *op);
void batch_norm_op_backward_input_only(batch_norm_op *op);

void calloc_batchnorm_weights(batch_norm_op *op);
void free_batchnorm_weights(batch_norm_op *op);

void calloc_batchnorm_dweights(batch_norm_op *op);
void free_batchnorm_dweights(batch_norm_op *op);

void load_batchnorm_weights(batch_norm_op *op, const float *gamma_array, const float *beta_array);
void save_batchnorm_weights(batch_norm_op *op);
