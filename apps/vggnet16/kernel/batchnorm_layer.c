//
// File:        batchnorm_layer.c
// Description: Implementation of batch normalization layer
//

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
#include "utils.h"

_Float16 bn1_avg_buf[BN1_CHANNELS];
_Float16 bn2_avg_buf[BN2_CHANNELS];
_Float16 bn3_avg_buf[BN3_CHANNELS];
_Float16 bn4_avg_buf[BN4_CHANNELS];
_Float16 bn5_avg_buf[BN5_CHANNELS];

_Float16 bn1_var_buf[BN1_CHANNELS];
_Float16 bn2_var_buf[BN2_CHANNELS];
_Float16 bn3_var_buf[BN3_CHANNELS];
_Float16 bn4_var_buf[BN4_CHANNELS];
_Float16 bn5_var_buf[BN5_CHANNELS];

// Running (inference-time) statistics. Initialised to mean 0 / var 1 by
// batchnorm_reset_running_stats(); updated as an EMA during training.
_Float16 bn1_run_mean_buf[BN1_CHANNELS];
_Float16 bn2_run_mean_buf[BN2_CHANNELS];
_Float16 bn3_run_mean_buf[BN3_CHANNELS];
_Float16 bn4_run_mean_buf[BN4_CHANNELS];
_Float16 bn5_run_mean_buf[BN5_CHANNELS];

_Float16 bn1_run_var_buf[BN1_CHANNELS];
_Float16 bn2_run_var_buf[BN2_CHANNELS];
_Float16 bn3_run_var_buf[BN3_CHANNELS];
_Float16 bn4_run_var_buf[BN4_CHANNELS];
_Float16 bn5_run_var_buf[BN5_CHANNELS];

_Float16 bn1_x_norm_buf[ALEXNET_STATIC_MAX_BATCH * BN1_UNITS];
_Float16 bn2_x_norm_buf[ALEXNET_STATIC_MAX_BATCH * BN2_UNITS];
_Float16 bn3_x_norm_buf[ALEXNET_STATIC_MAX_BATCH * BN3_UNITS];
_Float16 bn4_x_norm_buf[ALEXNET_STATIC_MAX_BATCH * BN4_UNITS];
_Float16 bn5_x_norm_buf[ALEXNET_STATIC_MAX_BATCH * BN5_UNITS];

// Scratch sized for the largest BN layer (bn1/bn2: C1 channels × 32×32 spatial)
_Float16 bn_dxnorm_scratch[ALEXNET_STATIC_MAX_BATCH * BN_MAX_UNITS];

static void bind_bn_layer_buffers(batch_norm_op *op)
{
    switch (op->layer_id) {
        case 1:
            op->avg    = bn1_avg_buf;
            op->var    = bn1_var_buf;
            op->x_norm = bn1_x_norm_buf;
            op->running_mean = bn1_run_mean_buf;
            op->running_var  = bn1_run_var_buf;
            break;
        case 2:
            op->avg    = bn2_avg_buf;
            op->var    = bn2_var_buf;
            op->x_norm = bn2_x_norm_buf;
            op->running_mean = bn2_run_mean_buf;
            op->running_var  = bn2_run_var_buf;
            break;
        case 3:
            op->avg    = bn3_avg_buf;
            op->var    = bn3_var_buf;
            op->x_norm = bn3_x_norm_buf;
            op->running_mean = bn3_run_mean_buf;
            op->running_var  = bn3_run_var_buf;
            break;
        case 4:
            op->avg    = bn4_avg_buf;
            op->var    = bn4_var_buf;
            op->x_norm = bn4_x_norm_buf;
            op->running_mean = bn4_run_mean_buf;
            op->running_var  = bn4_run_var_buf;
            break;
        case 5:
            op->avg    = bn5_avg_buf;
            op->var    = bn5_var_buf;
            op->x_norm = bn5_x_norm_buf;
            op->running_mean = bn5_run_mean_buf;
            op->running_var  = bn5_run_var_buf;
            break;
        default:
            printf_("Error: invalid batchnorm layer_id=%d\n", op->layer_id);
            exit(1);
    }
}


void batch_norm_op_forward(batch_norm_op *op)
{
    register _Float16 *input  = op->input;
    register _Float16 *output = op->output;

    bind_bn_layer_buffers(op);
    if (op->batchsize > ALEXNET_STATIC_MAX_BATCH) {
        printf_("Error: BN batchsize %d exceeds static max %d\n", op->batchsize, ALEXNET_STATIC_MAX_BATCH);
        exit(1);
    }

    // Inference: normalise with the frozen running stats. No batch statistics
    // are computed and no x_norm is stored (there is no backward pass at eval),
    // so a sample's output is independent of the rest of its batch.
    if (!op->is_training) {
        for (int n = 0; n < op->batchsize; n++) {
            for (int c = 0; c < op->channels; c++) {
                _Float16 inv_std = 1.0f / sqrtf(op->running_var[c] + EPSILON);
                _Float16 scale = op->gamma[c] * inv_std;
                _Float16 shift = op->beta[c] - (op->running_mean[c] * scale);
                
                unsigned long int s = op->spatial_size;

                while (s > 0) {
                    unsigned long int vl;

                    asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(s));

                    asm volatile("vle16.v v8, (%0)" : : "r"(input));

                    asm volatile("vfmv.v.f v16, %0" : : "f"(shift));

                    asm volatile("vfmacc.vf v16, %0, v8" : : "f"(scale));

                    asm volatile("vse16.v v16, (%0)" : : "r"(output));


                    input += vl;
                    output += vl;
                    s -= vl;
                }
            }
        }
        return;
    }

    memset_vectorized_zero_f32(op->avg, (size_t)op->channels);
    memset_vectorized_zero_f32(op->var, (size_t)op->channels);

    for (int p = 0; p < op->batchsize; p++) {
        for (int c = 0; c < op->channels; c++) {
            unsigned long int s = op->spatial_size;
            _Float16 sum = 0.0f;
            size_t vlmax_s;
            asm volatile("vsetvli %0, zero, e16, m8, ta, ma" : "=r"(vlmax_s));
            asm volatile("vfmv.s.f v16, %0" : : "f"(sum));

            while (s > 0) {
                unsigned long int vl;
                asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(s));
                asm volatile("vle16.v v8, (%0)" : : "r"(input));
                
                // Unordered Float Reduction Sum: v16 = sum(v8) + v16
                asm volatile("vfredusum.vs v16, v8, v16");
                input += vl;
                s -= vl;
            }
            asm volatile("vfmv.f.s %0, v16" : "=f"(sum));
            
            op->avg[c] += sum;
        }
    }
    
    _Float16 factor = 1.0f / (op->batchsize * op->spatial_size);
    int c_count = op->channels;
    _Float16 *avg_ptr = op->avg;

    while (c_count > 0) {
        unsigned long int vl;
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(c_count));
        
        asm volatile("vle16.v v8, (%0)" : : "r"(avg_ptr));
        asm volatile("vfmul.vf v8, v8, %0" : : "f"(factor));
        asm volatile("vse16.v v8, (%0)" : : "r"(avg_ptr));
        
        avg_ptr += vl;
        c_count -= vl;
    }

    // The mean pass advanced `input` to the end of the tensor; restart it before
    // the variance pass (the scalar version reset its offset here).
    input = op->input;

    for (int p = 0; p < op->batchsize; p++) {
        for (int c = 0; c < op->channels; c++) {
            unsigned long int s = op->spatial_size;
            _Float16 var_sum = 0.0f;
            _Float16 mean = op->avg[c];
            
            asm volatile("vfmv.s.f v16, %0" : : "f"(var_sum));

            while (s > 0) {
                unsigned long int vl;
                asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(s));
                asm volatile("vle16.v v8, (%0)" : : "r"(input));
                
                asm volatile("vfsub.vf v8, v8, %0" : : "f"(mean));
                asm volatile("vfmul.vv v8, v8, v8");
                asm volatile("vfredusum.vs v16, v8, v16");
                
                input += vl;
                s -= vl;
            }
            
            asm volatile("vfmv.f.s %0, v16" : "=f"(var_sum));
            
            op->var[c] += var_sum;
        }
    }
    
    _Float16 *var_ptr = op->var;

    // Update the running stats from this batch (EMA). PyTorch stores the
    // UNBIASED batch variance in running_var, so apply Bessel's correction
    // N/(N-1); the batch is still normalised below with the biased var above.
    _Float16 N = (_Float16)(op->batchsize * op->spatial_size);
    _Float16 bessel = (N > 1.0f) ? N / (N - 1.0f) : 1.0f;

    _Float16 momentum = BN_MOMENTUM;
    _Float16 one_minus_mom = 1.0f - momentum;
    _Float16 mom_x_bessel = momentum * bessel;

    // `factor` (= 1/N) is already computed above and reused here.
    // Reset the per-channel cursors: the mean loop advanced avg_ptr to the end.
    c_count = op->channels;
    avg_ptr = op->avg;
    _Float16 *rm_ptr = op->running_mean;
    _Float16 *rv_ptr = op->running_var;

    while (c_count > 0) {
        unsigned long int vl;
        asm volatile("vsetvli %0, %1, e16, m4, ta, ma" : "=r"(vl) : "r"(c_count));

        asm volatile("vle16.v v4, (%0)" : : "r"(avg_ptr));
        asm volatile("vle16.v v8, (%0)" : : "r"(var_ptr));
        asm volatile("vle16.v v12, (%0)" : : "r"(rm_ptr));
        asm volatile("vle16.v v16, (%0)" : : "r"(rv_ptr));

        asm volatile("vfmul.vf v8, v8, %0" : : "f"(factor));
        asm volatile("vse16.v v8, (%0)" : : "r"(var_ptr));

        asm volatile("vfmul.vf v12, v12, %0" : : "f"(one_minus_mom));
        asm volatile("vfmacc.vf v12, %0, v4" : : "f"(momentum));

        asm volatile("vfmul.vf v16, v16, %0" : : "f"(one_minus_mom));
        asm volatile("vfmacc.vf v16, %0, v8" : : "f"(mom_x_bessel)); 

        asm volatile("vse16.v v12, (%0)" : : "r"(rm_ptr));
        asm volatile("vse16.v v16, (%0)" : : "r"(rv_ptr));

        avg_ptr += vl;
        var_ptr += vl;
        rm_ptr += vl;
        rv_ptr += vl;
        c_count -= vl;
    }

    // Normalise + scale/shift (vectorized below). Restart from the base
    // pointers: the mean/variance passes advanced `input` to the end.
    _Float16 *in_ptr    = op->input;
    _Float16 *xnorm_ptr = op->x_norm;
    _Float16 *out_ptr   = op->output;

    for (int n = 0; n < op->batchsize; n++) {
        for (int cc = 0; cc < op->channels; cc++) {
            
            _Float16 mean = op->avg[cc];
            _Float16 inv_std = 1.0f / sqrtf(op->var[cc] + EPSILON);
            _Float16 g = op->gamma[cc];
            _Float16 b = op->beta[cc];
            
            unsigned long int s = op->spatial_size;

            while (s > 0) {
                unsigned long int vl;
                asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(s));
                
                asm volatile("vle16.v v8, (%0)" : : "r"(in_ptr));
                
                asm volatile("vfsub.vf v8, v8, %0" : : "f"(mean));
                asm volatile("vfmul.vf v8, v8, %0" : : "f"(inv_std));
                
                asm volatile("vse16.v v8, (%0)" : : "r"(xnorm_ptr));
                
                asm volatile("vfmul.vf v16, v8, %0" : : "f"(g));
                asm volatile("vfadd.vf v16, v16, %0" : : "f"(b));
                
                asm volatile("vse16.v v16, (%0)" : : "r"(out_ptr));
                
                in_ptr += vl;
                xnorm_ptr += vl;
                out_ptr += vl;
                s -= vl;
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
    int channels     = op->channels;
    int spatial_size = op->spatial_size;

    memset_vectorized_zero_f32(op->d_gamma, (size_t)channels);
    memset_vectorized_zero_f32(op->d_beta,  (size_t)channels);


    // dL/dgamma_c = sum over batch AND spatial of (d_output * x_norm). No 1/B
    // pass follows: cross_entropy_loss already scaled d_output by 1/batchsize,
    // so the sum is the mean gradient. Note this is a plain sum over spatial
    // too — dividing by M (= batchsize * spatial) would shrink gamma/beta by a
    // further factor of spatial_size, which for the conv BNs (spatial 1024 /
    // 256) leaves them effectively frozen.
    _Float16 *xnorm_ptr = op->x_norm;
    _Float16 *dout_ptr  = op->d_output;

    for (int n = 0; n < op->batchsize; n++) {
        for (int c = 0; c < channels; c++) {
            unsigned long int s = spatial_size;
            
            _Float16 dg_sum = 0.0f;
            _Float16 db_sum = 0.0f;

            asm volatile("vfmv.s.f v24, %0" : : "f"(dg_sum)); // Accumulator  d_gamma
            asm volatile("vfmv.s.f v25, %0" : : "f"(db_sum)); // Accumulator  d_beta

            while (s > 0) {
                unsigned long int vl;
                asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(s));
                
                asm volatile("vle16.v v0, (%0)" : : "r"(xnorm_ptr));
                asm volatile("vle16.v v8, (%0)" : : "r"(dout_ptr));
                
                asm volatile("vfredusum.vs v25, v8, v25");
                
                asm volatile("vfmul.vv v0, v0, v8");
                
                asm volatile("vfredusum.vs v24, v0, v24");
                
                xnorm_ptr += vl;
                dout_ptr += vl;
                s -= vl;
            }
            
            asm volatile("vfmv.f.s %0, v24" : "=f"(dg_sum));
            asm volatile("vfmv.f.s %0, v25" : "=f"(db_sum));
            
            op->d_gamma[c] += dg_sum;
            op->d_beta[c] += db_sum;
        }
    }

    // Restart the cursors: the d_gamma/d_beta pass advanced them to the end.
    xnorm_ptr = op->x_norm;
    dout_ptr  = op->d_output;
    _Float16 *din_ptr = op->d_input;

    // d_gamma/d_beta are now plain sums over batch AND spatial (the 1/batchsize
    // lives in cross_entropy_loss). The dx term needs them divided by
    // M = batchsize*spatial, which is what the scalar reference's  does.
    // Using 1/spatial here would leave dx a factor of batchsize too large --
    // and because this layer's dx feeds the next layer down, that error
    // compounds once per BN in the stack.
    _Float16 inv_M = 1.0f / ((_Float16)op->batchsize * (_Float16)spatial_size);

    for (int n = 0; n < op->batchsize; n++) {
        for (int c = 0; c < channels; c++) {
            
            _Float16 inv_std = 1.0f / sqrtf(op->var[c] + EPSILON);
            _Float16 factor1 = inv_std * op->gamma[c];            // (inv_std * gamma_c)
            _Float16 factor2 = op->d_beta[c] * inv_M;            // (d_beta / M)
            _Float16 factor3 = op->d_gamma[c] * inv_M;           // (d_gamma / M)
            
            unsigned long int s = spatial_size;

            while (s > 0) {
                unsigned long int vl;
                asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(s));
                
                asm volatile("vle16.v v8, (%0)"  : : "r"(xnorm_ptr));
                asm volatile("vle16.v v16, (%0)" : : "r"(dout_ptr));
                
                asm volatile("vfmul.vf v24, v8, %0" : : "f"(factor3));
                
                asm volatile("vfadd.vf v24, v24, %0" : : "f"(factor2));
                
                asm volatile("vfsub.vv v16, v16, v24");
                
                asm volatile("vfmul.vf v16, v16, %0" : : "f"(factor1));
                
                asm volatile("vse16.v v16, (%0)" : : "r"(din_ptr));
                
                xnorm_ptr += vl;
                dout_ptr  += vl;
                din_ptr   += vl;
                s -= vl;
            }
        }
    }
}
void batch_norm_op_backward_input_only(batch_norm_op *op)
{
    int channels     = op->channels;
    int spatial_size = op->spatial_size;
    _Float16 M = (_Float16)(op->batchsize * spatial_size);

    _Float16 S1[BN_MAX_CHANNELS];
    _Float16 S2[BN_MAX_CHANNELS];
    memset_vectorized_zero_f32(S1, (size_t)channels);
    memset_vectorized_zero_f32(S2, (size_t)channels);

    _Float16 *xnorm_ptr = op->x_norm;
    _Float16 *dout_ptr  = op->d_output;

    for (int n = 0; n < op->batchsize; n++) {
        for (int c = 0; c < channels; c++) {
            unsigned long int s = spatial_size;
            _Float16 sum_dy = 0.0f;
            _Float16 sum_dy_xn = 0.0f;

            asm volatile("vfmv.s.f v24, %0" : : "f"(sum_dy_xn)); 
            asm volatile("vfmv.s.f v25, %0" : : "f"(sum_dy)); 

            while (s > 0) {
                unsigned long int vl;
                asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(s));
                
                asm volatile("vle16.v v0, (%0)" : : "r"(xnorm_ptr));
                asm volatile("vle16.v v8, (%0)" : : "r"(dout_ptr));
                
                asm volatile("vfredusum.vs v25, v8, v25");
                
                asm volatile("vfmul.vv v0, v0, v8");
                
                asm volatile("vfredusum.vs v24, v0, v24");
                
                xnorm_ptr += vl;
                dout_ptr += vl;
                s -= vl;
            }
            
            asm volatile("vfmv.f.s %0, v24" : "=f"(sum_dy_xn));
            asm volatile("vfmv.f.s %0, v25" : "=f"(sum_dy));
            
            S1[c] += sum_dy;
            S2[c] += sum_dy_xn;
        }
    }

    xnorm_ptr = op->x_norm;
    dout_ptr  = op->d_output;
    _Float16 *din_ptr   = op->d_input;
    _Float16 inv_M = 1.0f / M;

    for (int n = 0; n < op->batchsize; n++) {
        for (int c = 0; c < channels; c++) {
            
            _Float16 inv_std = 1.0f / sqrtf(op->var[c] + EPSILON);
            _Float16 factor1 = inv_std * op->gamma[c]; 
            _Float16 factor2 = S1[c] * inv_M; //   d_beta / spatial_size
            _Float16 factor3 = S2[c] * inv_M; //   d_gamma / spatial_size

            unsigned long int s = spatial_size;

            while (s > 0) {
                unsigned long int vl;
                asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(s));
                
                asm volatile("vle16.v v8, (%0)"  : : "r"(xnorm_ptr));
                asm volatile("vle16.v v16, (%0)" : : "r"(dout_ptr));
                
                asm volatile("vfmul.vf v24, v8, %0" : : "f"(factor3));
                
                asm volatile("vfadd.vf v24, v24, %0" : : "f"(factor2));
                
                asm volatile("vfsub.vv v16, v16, v24");
                
                asm volatile("vfmul.vf v16, v16, %0" : : "f"(factor1));
                
                asm volatile("vse16.v v16, (%0)" : : "r"(din_ptr));
                
                xnorm_ptr += vl;
                dout_ptr  += vl;
                din_ptr   += vl;
                s -= vl;
            }
        }
    }
}


void calloc_batchnorm_weights(batch_norm_op *op)
{
    if (op->gamma)
        memset_vectorized_zero_f32(op->gamma, (size_t)op->channels);
    if (op->beta)
        memset_vectorized_zero_f32(op->beta, (size_t)op->channels);
}

void free_batchnorm_weights(batch_norm_op *op)
{
    (void)op;
}

void calloc_batchnorm_dweights(batch_norm_op *op)
{
    if (op->d_gamma)
        memset_vectorized_zero_f32(op->d_gamma, (size_t)op->channels);
    if (op->d_beta)
        memset_vectorized_zero_f32(op->d_beta, (size_t)op->channels);
}

void free_batchnorm_dweights(batch_norm_op *op)
{
    (void)op;
}

void save_batchnorm_weights(batch_norm_op *op)
{
    (void)op;
}

void load_batchnorm_weights(batch_norm_op *op, const _Float16 *gamma_array, const _Float16 *beta_array)
{
    memcpy_vectorized_f32(op->gamma, gamma_array, (size_t)op->channels);
    memcpy_vectorized_f32(op->beta,  beta_array,  (size_t)op->channels);
}

void batchnorm_reset_running_stats(void)
{
    struct { _Float16 *mean; _Float16 *var; int n; } bn[] = {
        { bn1_run_mean_buf, bn1_run_var_buf, BN1_CHANNELS },
        { bn2_run_mean_buf, bn2_run_var_buf, BN2_CHANNELS },
        { bn3_run_mean_buf, bn3_run_var_buf, BN3_CHANNELS },
        { bn4_run_mean_buf, bn4_run_var_buf, BN4_CHANNELS },
        { bn5_run_mean_buf, bn5_run_var_buf, BN5_CHANNELS },
    };

    _Float16 zero = 0.0f;
    _Float16 one  = 1.0f;

    for (int k = 0; k < 5; k++) {
        int c_count  = bn[k].n;
        _Float16 *m_ptr = bn[k].mean;
        _Float16 *v_ptr = bn[k].var;

        while (c_count > 0) {
            unsigned long int vl;
            asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(c_count));

            asm volatile("vfmv.v.f v8, %0" : : "f"(zero));
            
            asm volatile("vfmv.v.f v16, %0" : : "f"(one));

            asm volatile("vse16.v v8, (%0)" : : "r"(m_ptr));
            asm volatile("vse16.v v16, (%0)" : : "r"(v_ptr));

            m_ptr += vl;
            v_ptr += vl;
            c_count -= vl;
        }
    }
}