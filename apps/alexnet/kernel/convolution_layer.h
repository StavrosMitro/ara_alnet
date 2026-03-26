//
// File:        convolution_layer.h
// Description: interface of convolution layer
// Author:      Haris Wang
//
// #include <stdlib.h>

#ifndef ALEXNET_STATIC_MAX_BATCH
#define ALEXNET_STATIC_MAX_BATCH 400
#endif

#define CONV_MAX_IKK  (256 * 3 * 3)
#define CONV_MAX_OC   256
#define CONV_MAX_OWOH (32 * 32)
#define CONV_MAX_DXCOL (CONV_MAX_IKK * CONV_MAX_OWOH)
#define CONV_MAX_DOCOPY (CONV_MAX_OC * CONV_MAX_OWOH)
#define CONV_MAX_INTERNAL ((CONV_MAX_IKK + 7) / 8)
#define CONV_MAX_T_DWEIGHTS  (CONV_MAX_OC * CONV_MAX_IKK)
#define CONV_MAX_T_INPUT_COL (CONV_MAX_OWOH * CONV_MAX_IKK)
#define CONV5_IKK (256 * 3 * 3)
#define CONV5_OWOH (8 * 8)
#define CONV5_INPUT_COL_SIZE (ALEXNET_STATIC_MAX_BATCH * CONV5_IKK * CONV5_OWOH)

typedef struct conv_op {
    float *input;   float *d_input;
    float *output;  float *d_output;
    float *weights; float *d_weights;
    float *bias;    float *d_bias;
    float *input_col;

    int in_channels, out_channels;
    int kernel_size; int padding; int stride;
    int in_w, in_h, out_w, out_h;
    int in_units, out_units;

    short batchsize;
    short layer_id;
} conv_op;

// typedef struct conv_args{
//     conv_op *op;
//     short batch_id;
//     short st_tunits;
//     short ed_tunits;
// } conv_args;



void conv_op_forward(conv_op *op);
void conv_op_backward(conv_op *op);
void conv_op_backward_full(conv_op *op);
void conv_op_backward_input_only(conv_op *op);

inline void calloc_conv_weights(conv_op *op);
inline void free_conv_weights(conv_op *op);

inline void calloc_conv_dweights(conv_op *op);
inline void free_conv_dweights(conv_op *op);

inline void load_conv_weights(conv_op *op, float *w_array, float *b_array);
inline void save_conv_weights(conv_op *op);
