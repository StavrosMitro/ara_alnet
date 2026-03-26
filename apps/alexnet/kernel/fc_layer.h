//
// File:        fc_layer.h
// Description: interface of full connected layer
// Author:      Haris Wang
//
// #include <stdlib.h>

#ifndef ALEXNET_STATIC_MAX_BATCH
#define ALEXNET_STATIC_MAX_BATCH 400
#endif

#define FC_MAX_IN_UNITS 2048
#define FC_MAX_INTERNAL 64

typedef struct fc_op {
    float *input;   float *d_input;
    float *output;  float *d_output;
    float *weights; float *d_weights;
    float *bias;    float *d_bias;
    int in_units, out_units;

    short batchsize;
    short layer_id;
} fc_op;


void fc_op_forward(fc_op *op);
void fc_op_backward(fc_op *op);
void fc_op_backward_full(fc_op *op);
void fc_op_backward_input_only(fc_op *op);

inline void calloc_fc_weights(fc_op *op);
inline void free_fc_weights(fc_op *op);

inline void calloc_fc_dweights(fc_op *op);
inline void free_fc_dweights(fc_op *op);

inline void load_fc_weights(fc_op *op, float *w_array, float *b_array);
inline void save_fc_weights(fc_op *op );
