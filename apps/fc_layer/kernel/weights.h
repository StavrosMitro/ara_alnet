#ifndef WEIGHTS_H
#define WEIGHTS_H

// From weights.c
extern float fc1_weights[1048576];
extern float fc1_bias[512];

// From the assembly file (.incbin)
extern const float test_inputs[];
extern const float test_targets[];
extern const int test_labels[];

#endif