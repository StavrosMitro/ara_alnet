//
// File:        batchnorm_layer.c
// Description: Implementation of batch normalization layer
// Author:


#include <stdlib.h>
#include <math.h>
#include <string.h>
#ifdef SPIKE
#include <printf.h>
#elif defined ARA_LINUX
#include <stdio.h>
#else
#include "printf.h"
#endif
#include "batchnorm_layer.h"

float bn1_avg_buf[BN1_CHANNELS];
float bn2_avg_buf[BN2_CHANNELS];
float bn3_avg_buf[BN3_CHANNELS];
float bn4_avg_buf[BN4_CHANNELS];
float bn5_avg_buf[BN5_CHANNELS];

float bn1_var_buf[BN1_CHANNELS];
float bn2_var_buf[BN2_CHANNELS];
float bn3_var_buf[BN3_CHANNELS];
float bn4_var_buf[BN4_CHANNELS];
float bn5_var_buf[BN5_CHANNELS];

float bn1_x_norm_buf[ALEXNET_STATIC_MAX_BATCH * BN1_UNITS];
float bn2_x_norm_buf[ALEXNET_STATIC_MAX_BATCH * BN2_UNITS];
float bn3_x_norm_buf[ALEXNET_STATIC_MAX_BATCH * BN3_UNITS];
float bn4_x_norm_buf[ALEXNET_STATIC_MAX_BATCH * BN4_UNITS];
float bn5_x_norm_buf[ALEXNET_STATIC_MAX_BATCH * BN5_UNITS];

float bn_dxnorm_scratch[ALEXNET_STATIC_MAX_BATCH * BN3_UNITS];

static void bind_bn_layer_buffers(batch_norm_op *op)
{
    switch (op->layer_id) {
        case 1:
            op->avg = bn1_avg_buf;
            op->var = bn1_var_buf;
            op->x_norm = bn1_x_norm_buf;
            break;
        case 2:
            op->avg = bn2_avg_buf;
            op->var = bn2_var_buf;
            op->x_norm = bn2_x_norm_buf;
            break;
        case 3:
            op->avg = bn3_avg_buf;
            op->var = bn3_var_buf;
            op->x_norm = bn3_x_norm_buf;
            break;
        case 4:
            op->avg = bn4_avg_buf;
            op->var = bn4_var_buf;
            op->x_norm = bn4_x_norm_buf;
            break;
        case 5:
            op->avg = bn5_avg_buf;
            op->var = bn5_var_buf;
            op->x_norm = bn5_x_norm_buf;
            break;
        default:
            printf_("Error: invalid batchnorm layer_id=%d\n", op->layer_id);
            exit(1);
    }
}


/*

typedef struct batch_norm_op {
    float *input; float *d_input;
    float *output; float *d_output;
    float *gamma; float *d_gamma; --> per channel
    float *beta; float *d_beta; --> per channel

    int units;
    int channels;
    int spatial_size;
    short batchsize;
    
    float *x_norm;
    float *avg;
    float *var;
} batch_norm_op;
 
*/



void batch_norm_op_forward(batch_norm_op *op)
{
    register float *input  = op->input;
    register float *output = op->output;

    bind_bn_layer_buffers(op);
    if (op->batchsize > ALEXNET_STATIC_MAX_BATCH) {
        printf_("Error: BN batchsize %d exceeds static max %d\n", op->batchsize, ALEXNET_STATIC_MAX_BATCH);
        exit(1);
    }
    memset(op->avg, 0, (size_t)op->channels * sizeof(float));
    memset(op->var, 0, (size_t)op->channels * sizeof(float));

    register int i, p, c;

    // calculate mean for each unit along batch axis
    // [image_0: unit_0, unit_1, ..., unit_N, image_1: unit_0, ..., image_B: ..., unit_N]
    //batch_idx * (C*H*W) + c*(H*W) + h*W + w

    register int offset = 0;

    for (p=0; p< op->batchsize; p++) {
        for (c=0; c < op->channels; c++) {
            for (i=0; i < op-> spatial_size; i++) {
                op->avg[c] += input[offset++];
            }
        }
    }
    offset=0;
    register float factor = 1.0f / (op->batchsize * op->spatial_size);
    register float diff = 0;
    for (c = 0; c < op->channels; c++)
        op->avg[c] *= factor;

    for (p=0; p< op->batchsize; p++) {
        for (c=0; c < op->channels; c++) {
            for (i=0; i < op-> spatial_size; i++) {
                diff = input[offset++] - op->avg[c];
                op->var[c] += diff * diff;
            }
        }
    }

    for (c = 0; c < op->channels; c++)
        op->var[c] *= factor;

    offset = 0;
    register float inv_std;

    for (int n = 0; n < op->batchsize; n++) {
        for (int c = 0; c < op->channels; c++) {
            inv_std = 1.0f / sqrtf(op->var[c] + EPSILON);
            for (int s = 0; s < op->spatial_size; s++) {
                op->x_norm[offset] = (input[offset] - op->avg[c]) * inv_std;

                output[offset] = op->gamma[c] * op->x_norm[offset]  + op->beta[c];
                offset++;
            }
        }
    }
}


void batch_norm_op_backward(batch_norm_op *op)
{
    batch_norm_op_backward_full(op);
}

void batch_norm_op_backward_full(batch_norm_op *op)
{
    int channels = op->channels;
    int spatial_size = op->spatial_size;
    int total_elements = op->batchsize * channels * spatial_size;
    float M = (float)(op->batchsize * spatial_size);

    // calculate delta_gamma: d_gamma[c] = (1/B) * sum_{n,s} x_norm_{n,c,s} * d_output_{n,c,s}
    for (int c = 0; c < channels; c++) {
        op->d_gamma[c] = 0.0f;
        op->d_beta[c] = 0.0f;
    }

    register int offset = 0;
    for (int n = 0; n < op->batchsize; n++) {
        for (int c = 0; c < channels; c++) {
            for (int s = 0; s < spatial_size; s++) {
                op->d_gamma[c] += op->x_norm[offset] * op->d_output[offset];
                offset++;
            }
        }
    }
    for (int c = 0; c < channels; c++)
        op->d_gamma[c] /= M;

    // calculate delta_beta: d_beta[c] = (1/B) * sum_{n,s} d_output_{n,c,s}
    offset = 0;
    for (int n = 0; n < op->batchsize; n++) {
        for (int c = 0; c < channels; c++) {
            for (int s = 0; s < spatial_size; s++) {
                op->d_beta[c] += op->d_output[offset++];
            }
        }
    }
    for (int c = 0; c < channels; c++)
        op->d_beta[c] /= M;

    // calculate d_input (per-sample)
    float *dxnorm = bn_dxnorm_scratch;
    float S1[BN3_CHANNELS] = {0};
    float S2[BN3_CHANNELS] = {0};

    offset = 0;
    for (int n = 0; n < op->batchsize; n++) {
        for (int c = 0; c < channels; c++) {
            for (int s = 0; s < spatial_size; s++) {
                dxnorm[offset] = op->d_output[offset] * op->gamma[c];
                offset++;
            }
        }
    }

    offset = 0;
    for (int n = 0; n < op->batchsize; n++) {
        for (int c = 0; c < channels; c++) {
            for (int s = 0; s < spatial_size; s++) {
                float dxn = dxnorm[offset];
                float xn = op->x_norm[offset];
                S1[c] += dxn;
                S2[c] += dxn * xn;
                offset++;
            }
        }
    }

    offset = 0;
    for (int n = 0; n < op->batchsize; n++) {
        for (int c = 0; c < channels; c++) {
            float inv_std = 1.0f / sqrtf(op->var[c] + EPSILON);
            float s1_c = S1[c];
            float s2_c = S2[c];
            for (int s = 0; s < spatial_size; s++) {
                float dxn = dxnorm[offset];
                float xn = op->x_norm[offset];
                float term = (s1_c + xn * s2_c) / M;
                op->d_input[offset] = inv_std * (dxn - term);
                offset++;
            }
        }
    }

    (void)dxnorm;
}

void batch_norm_op_backward_input_only(batch_norm_op *op)
{
    int channels = op->channels;
    int spatial_size = op->spatial_size;
    int total_elements = op->batchsize * channels * spatial_size;
    float M = (float)(op->batchsize * spatial_size);

    // Only compute d_input for frozen batchnorm layers.
    float *dxnorm = bn_dxnorm_scratch;
    float S1[BN3_CHANNELS] = {0};
    float S2[BN3_CHANNELS] = {0};

    int offset = 0;
    for (int n = 0; n < op->batchsize; n++) {
        for (int c = 0; c < channels; c++) {
            for (int s = 0; s < spatial_size; s++) {
                dxnorm[offset] = op->d_output[offset] * op->gamma[c];
                offset++;
            }
        }
    }

    offset = 0;
    for (int n = 0; n < op->batchsize; n++) {
        for (int c = 0; c < channels; c++) {
            for (int s = 0; s < spatial_size; s++) {
                float dxn = dxnorm[offset];
                float xn = op->x_norm[offset];
                S1[c] += dxn;
                S2[c] += dxn * xn;
                offset++;
            }
        }
    }

    offset = 0;
    for (int n = 0; n < op->batchsize; n++) {
        for (int c = 0; c < channels; c++) {
            float inv_std = 1.0f / sqrtf(op->var[c] + EPSILON);
            float s1_c = S1[c];
            float s2_c = S2[c];
            for (int s = 0; s < spatial_size; s++) {
                float dxn = dxnorm[offset];
                float xn = op->x_norm[offset];
                float term = (s1_c + xn * s2_c) / M;
                op->d_input[offset] = inv_std * (dxn - term);
                offset++;
            }
        }
    }

    (void)dxnorm;
}


void calloc_batchnorm_weights(batch_norm_op *op)
{
    if (op->gamma)
        memset(op->gamma, 0, (size_t)op->channels * sizeof(float));
    if (op->beta)
        memset(op->beta, 0, (size_t)op->channels * sizeof(float));
}

void free_batchnorm_weights(batch_norm_op *op)
{
    (void)op;
}

void calloc_batchnorm_dweights(batch_norm_op *op)
{
    if (op->d_gamma)
        memset(op->d_gamma, 0, (size_t)op->channels * sizeof(float));
    if (op->d_beta)
        memset(op->d_beta, 0, (size_t)op->channels * sizeof(float));
}

void free_batchnorm_dweights(batch_norm_op *op)
{
    (void)op;
}

void save_batchnorm_weights(batch_norm_op *op)
{
    (void)op;
}

void load_batchnorm_weights(batch_norm_op *op, const float *gamma_array, const float *beta_array)
{
    memcpy(op->gamma, gamma_array, op->channels * sizeof(float));
    memcpy(op->beta,  beta_array,  op->channels * sizeof(float));
}