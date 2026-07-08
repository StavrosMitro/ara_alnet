//
// File:        fc_layer.h
// Description: Pure FP16 fully connected layer interface
// Author:      Haris Wang
// FP16 refactor: Stavros Mitropoulos, NTUA
//

#include <stdint.h>

#ifndef ALEXNET_STATIC_MAX_BATCH
#define ALEXNET_STATIC_MAX_BATCH 2
#endif

#define FC_MAX_IN_UNITS  2048
#define FC_MAX_INTERNAL  512

// ---------------------------------------------------------------------------
// Operation descriptor — all data pointers are pure FP16
// ---------------------------------------------------------------------------
typedef struct fc_op {
    _Float16 *input;   _Float16 *d_input;
    _Float16 *output;  _Float16 *d_output;
    _Float16 *weights; _Float16 *d_weights;
    _Float16 *bias;    _Float16 *d_bias;
    int in_units, out_units;

    short batchsize;
    short layer_id;
} fc_op;

typedef struct fc_backward_cycle_breakdown {
    int64_t d_input_cycles;
    int64_t d_bias_cycles;
    int64_t d_weights_cycles;
} fc_backward_cycle_breakdown;

// ---------------------------------------------------------------------------
// Forward pass
// Sets frm=RNE, scales input ×(1/8) before fmatmul to prevent FP16 overflow.
// ---------------------------------------------------------------------------
void fc_op_forward(fc_op *op);

// ---------------------------------------------------------------------------
// Backward pass with dynamic loss scaling via fflags overflow detection.
// Returns 1 when the weight update was applied, 0 when the step was skipped
// (overflow detected — scale was halved, caller should not apply SGD externally).
// ---------------------------------------------------------------------------
int fc_op_backward(fc_op *op, fc_backward_cycle_breakdown *cycles);

// Gradient-only backward for frozen layers (no weight update).
void fc_op_backward_input_only(fc_op *op);

// Weight management
void calloc_fc_weights(fc_op *op);
void free_fc_weights(fc_op *op);
void calloc_fc_dweights(fc_op *op);
void free_fc_dweights(fc_op *op);
void load_fc_weights(fc_op *op, _Float16 *w_array, _Float16 *b_array);
void save_fc_weights(fc_op *op);
