#ifndef WEIGHTS_H
#define WEIGHTS_H

// From weights.c
extern float conv1_weights[16641];
extern float conv1_bias[43];

// From the assembly file (.incbin)
extern const float test_inputs[];
extern const float test_targets[];
extern const int test_labels[];

#endif