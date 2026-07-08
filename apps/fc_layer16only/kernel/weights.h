#ifndef WEIGHTS_H
#define WEIGHTS_H

// From weights.c
extern _Float16 fc1_weights_32[1048576];
extern _Float16 fc1_bias_32[512];

// From the assembly file (.incbin)
extern const _Float16 test_inputs_32[];
extern const _Float16 test_targets_32[];
extern const int test_labels_32[];

#endif