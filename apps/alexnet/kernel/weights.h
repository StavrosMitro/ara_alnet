#ifndef WEIGHTS_H
#define WEIGHTS_H

extern float conv1_weights[1728];
extern float conv1_bias[64];

extern float conv2_weights[73728];
extern float conv2_bias[128];

extern float conv3_weights[294912];
extern float conv3_bias[256];

extern float conv4_weights[589824];
extern float conv4_bias[256];

extern float conv5_weights[294912];
extern float conv5_bias[128];

extern float fc1_weights[1048576];
extern float fc1_bias[512];

extern float fc2_weights[262144];
extern float fc2_bias[512];

extern float fc3_weights[5120];
extern float fc3_bias[10];


extern float bn1_gamma[64];
extern float bn1_beta[64];
extern float bn1_mean[64];
extern float bn1_var[64];

extern float bn2_gamma[128];
extern float bn2_beta[128];
extern float bn2_mean[128];
extern float bn2_var[128];

extern float bn3_gamma[256];
extern float bn3_beta[256];
extern float bn3_mean[256];
extern float bn3_var[256];

extern float bn4_gamma[256];
extern float bn4_beta[256];
extern float bn4_mean[256];
extern float bn4_var[256];

extern float bn5_gamma[128];
extern float bn5_beta[128];
extern float bn5_mean[128];
extern float bn5_var[128];

#endif // WEIGHTS_H