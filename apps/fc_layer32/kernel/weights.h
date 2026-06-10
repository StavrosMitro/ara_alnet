#ifndef WEIGHTS_H
#define WEIGHTS_H

// From weights.c
extern float fc1_weights_32[1048576];
extern float fc1_bias_32[512];

// From the assembly file (.incbin)
extern const float test_inputs_32[];
extern const float test_targets_32[];
extern const int test_labels_32[];

#endif