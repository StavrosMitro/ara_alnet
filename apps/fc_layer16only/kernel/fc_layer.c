//
// File:        fc_layer.c
// Description: Pure FP16 fully connected layer
//              - Static input scaling (×1/8) in forward pass
//              - Dynamic loss scaling via RISC-V fflags CSR in backward pass
// Author:      Haris Wang
// FP16 refactor: Stavros Mitropoulos, NTUA
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
#include "fc_layer.h"
#include "matrix.h"
#include "runtime.h"
#include "fmatmul.h"

#if ALEXNET_STATIC_MAX_BATCH <= 4
#define FMATMUL_MAX_M 4
#elif ALEXNET_STATIC_MAX_BATCH <= 8
#define FMATMUL_MAX_M 8
#elif ALEXNET_STATIC_MAX_BATCH <= 64
#define FMATMUL_MAX_M (((ALEXNET_STATIC_MAX_BATCH + 15) / 16) * 16)
#elif ALEXNET_STATIC_MAX_BATCH <= 128
#define FMATMUL_MAX_M (((ALEXNET_STATIC_MAX_BATCH + 7) / 8) * 8)
#else
#define FMATMUL_MAX_M (((ALEXNET_STATIC_MAX_BATCH + 3) / 4) * 4)
#endif

#define FMATMUL_MAX_N FC_MAX_IN_UNITS
#define FMATMUL_MAX_K FC_MAX_IN_UNITS

// Scale input by 2^-3 before forward matmul (exact power-of-2: no rounding error).
// Keeps worst-case FP16 accumulator sum well below 65504.
#define INPUT_SCALE   ((_Float16)0.00390625f)

#define LEARNING_RATE ((_Float16)0.001f)

static inline unsigned long int fmatmul_row_block(unsigned long int m)
{
    if (m <= 4)   return 4;
    if (m <= 8)   return 8;
    if (m <= 64)  return 16;
    if (m <= 128) return 8;
    return 4;
}

static inline int64_t fc_cycle_count_local(void)
{
    int64_t cycle_count = 0;
    asm volatile("fence; csrr %0, cycle" : "=r"(cycle_count));
    return cycle_count;
}

static void matrix_multiply_scalar_fused(const _Float16 *a, const _Float16 *b,
                                         const _Float16 *bias, _Float16 *c,
                                         const int M, const int N, const int K);

static _Float16 fc_t_output_scratch[ALEXNET_STATIC_MAX_BATCH * FC_MAX_INTERNAL * 2];

// ---------------------------------------------------------------------------
// Forward pass
// ---------------------------------------------------------------------------
void fc_op_forward(fc_op *op)
{
    SET_FRM_RNE();

    _Float16 *fmatmul_a_scratch = shared_memory_pool_16;

    if (op->batchsize <= 0 || op->in_units <= 0 || op->out_units <= 0)
        return;

    unsigned long int block = fmatmul_row_block((unsigned long int)op->batchsize);
    unsigned long int padded_m = (((unsigned long int)op->batchsize + block - 1) / block) * block;

    if ((unsigned long int)op->in_units > FMATMUL_MAX_N ||
        (unsigned long int)op->out_units > FMATMUL_MAX_K ||
        padded_m > FMATMUL_MAX_M)
    {
        printf_("[SCALAR] fc_op_forward: batchsize=%d in=%d out=%d padded_m=%lu MAX_M=%d\n",
                op->batchsize, op->in_units, op->out_units, padded_m, FMATMUL_MAX_M);
        matrix_multiply_scalar_fused(op->input, op->weights, op->bias, op->output,
                                     op->batchsize, op->in_units, op->out_units);
        return;
    }

    const size_t mn  = (size_t)op->batchsize * (size_t)op->in_units;
    const size_t pnk = (size_t)padded_m      * (size_t)op->in_units;

    if (padded_m == (unsigned long int)op->batchsize) {
        // No padding needed: pass input/output buffers directly.
        // INPUT_SCALE is pre-absorbed into weights at load time.
        fmatmul_fused_16(op->output, op->input, op->weights, op->bias,
                         padded_m,
                         (unsigned long int)op->in_units,
                         (unsigned long int)op->out_units);
    } else {
        // Padding needed: fuse INPUT_SCALE into the copy, zero-pad the rest.
        size_t remaining_mn = mn;
        const _Float16 *src_mn = op->input;
        _Float16 *dst_mn = fmatmul_a_scratch;
        while (remaining_mn > 0)
        {
            size_t vl = 0;
            asm volatile("vsetvli %0, %1, e16, m1, ta, ma" : "=r"(vl) : "r"(remaining_mn));
            asm volatile("vle16.v v0, (%0);"       : : "r"(src_mn) : "memory");
            asm volatile("vfmul.vf v0, v0, %0"     : : "f"(INPUT_SCALE));
            asm volatile("vse16.v v0, (%0);"       : : "r"(dst_mn) : "memory");
            src_mn += vl;
            dst_mn += vl;
            remaining_mn -= vl;
        }

        size_t remaining_pad = pnk - mn;
        _Float16 *dst_pad = fmatmul_a_scratch + mn;
        while (remaining_pad > 0)
        {
            size_t vl = 0;
            asm volatile("vsetvli %0, %1, e16, m1, ta, ma" : "=r"(vl) : "r"(remaining_pad));
            asm volatile("vmv.v.x v0, zero");
            asm volatile("vse16.v v0, (%0);" : : "r"(dst_pad) : "memory");
            dst_pad += vl;
            remaining_pad -= vl;
        }

        fmatmul_fused_16(fmatmul_c_scratch_16, fmatmul_a_scratch, op->weights, op->bias,
                         padded_m,
                         (unsigned long int)op->in_units,
                         (unsigned long int)op->out_units);

        // Copy only the real batchsize rows of output back (discard padded rows).
        const size_t mk = (size_t)op->batchsize * (size_t)op->out_units;
        size_t remaining_mk = mk;
        const _Float16 *src_mk = fmatmul_c_scratch_16;
        _Float16 *dst_mk = op->output;
        while (remaining_mk > 0)
        {
            size_t vl = 0;
            asm volatile("vsetvli %0, %1, e16, m1, ta, ma" : "=r"(vl) : "r"(remaining_mk));
            asm volatile("vle16.v v0, (%0);" : : "r"(src_mk) : "memory");
            asm volatile("vse16.v v0, (%0);" : : "r"(dst_mk) : "memory");
            src_mk += vl;
            dst_mk += vl;
            remaining_mk -= vl;
        }
    }
}

// ---------------------------------------------------------------------------
// Backward pass with dynamic loss scaling
//
// 1. Scale d_output UP by dynamic_loss_scale (prevents gradient underflow).
// 2. CLEAR_FFLAGS.
// 3. Compute d_input, d_bias, d_weights (all operate on scaled gradients).
// 4. READ_FFLAGS.
// 5a. Overflow → halve scale, skip weight update, return 0.
// 5b. Clean  → unscale d_weights, SGD update, track steps; double scale
//              after 200 consecutive clean steps. Return 1.
// ---------------------------------------------------------------------------
int fc_op_backward(fc_op *op, fc_backward_cycle_breakdown *cycles)
{
    static _Float16 dynamic_loss_scale = (_Float16)1024.0f;
    static int      successful_steps   = 0;

    int64_t t0 = 0;

    if (cycles) {
        cycles->d_input_cycles   = 0;
        cycles->d_bias_cycles    = 0;
        cycles->d_weights_cycles = 0;
    }

    if (op->d_weights == NULL || op->d_bias == NULL) {
        fc_op_backward_input_only(op);
        return 0;
    }

    // Scale d_output up so small gradients survive FP16 underflow.
    const size_t d_out_elems = (size_t)op->batchsize * (size_t)op->out_units;
    vector_scale_fp16(op->d_output, dynamic_loss_scale, d_out_elems);

    // Arm the overflow detector — clear before all backward kernels.
    CLEAR_FFLAGS();

    // d_input = d_output * weights^T
    t0 = fc_cycle_count_local();
    matrix_multiply_nt_16(op->d_output, op->weights, op->d_input,
                          op->batchsize, op->out_units, op->in_units);
    if (cycles) cycles->d_input_cycles += fc_cycle_count_local() - t0;

    // d_bias = mean(d_output, axis=batch)
    t0 = fc_cycle_count_local();
    calc_bias_gradient_vec_16(op->d_bias, op->d_output, op->out_units, op->batchsize);
    if (cycles) cycles->d_bias_cycles += fc_cycle_count_local() - t0;

    // d_weights = input^T * d_output
    t0 = fc_cycle_count_local();
    register _Float16 *w_deltas = op->d_weights;
    matrix_multiply_tn_16(op->input, op->d_output, w_deltas,
                          op->batchsize, op->in_units, op->out_units);
    if (cycles) cycles->d_weights_cycles += fc_cycle_count_local() - t0;

    // Check whether any active-element vector FP op overflowed.
    unsigned int flags = 0;
    READ_FFLAGS(flags);

    if (flags & FFLAG_OVERFLOW_MASK) {
        // Gradient magnitude exceeded FP16 range — halve scale and skip update.
        dynamic_loss_scale *= (_Float16)0.5f;
        successful_steps    = 0;
        return 0;
    }

    // Unscale d_weights before SGD (d_input is passed to the previous layer as-is).
    const _Float16 inv_scale    = (_Float16)1.0f / dynamic_loss_scale;
    const size_t   weight_count = (size_t)op->in_units * (size_t)op->out_units;
    vector_scale_fp16(op->d_weights, inv_scale, weight_count);

    _Float16 lr_test = LEARNING_RATE;
    _Float16 dw_test = op->d_weights[0];
    _Float16 update_val = lr_test * dw_test;
    printf_("Test Update: dw=%f, update_val=%f\n", (float)dw_test, (float)update_val);
    // SGD weight update: w -= lr * dw  (vectorised via fmatmul.c helper)
    const size_t bias_count = (size_t)op->out_units;
    size_t rem;
    _Float16 *w_ptr  = op->weights;
    _Float16 *dw_ptr = op->d_weights;
    rem = weight_count;
    while (rem > 0) {
        size_t vl;
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(rem));
        asm volatile("vle16.v v8,  (%0)" :: "r"(w_ptr));
        asm volatile("vle16.v v16, (%0)" :: "r"(dw_ptr));
        asm volatile("vfnmsac.vf v8, %0, v16" :: "f"(LEARNING_RATE));
        asm volatile("vse16.v v8,  (%0)" :: "r"(w_ptr) : "memory");
        w_ptr += vl; dw_ptr += vl; rem -= vl;
    }

    w_ptr  = op->bias;
    dw_ptr = op->d_bias;
    rem = bias_count;
    while (rem > 0) {
        size_t vl;
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(rem));
        asm volatile("vle16.v v8,  (%0)" :: "r"(w_ptr));
        asm volatile("vle16.v v16, (%0)" :: "r"(dw_ptr));
        asm volatile("vfnmsac.vf v8, %0, v16" :: "f"(LEARNING_RATE));
        asm volatile("vse16.v v8,  (%0)" :: "r"(w_ptr) : "memory");
        w_ptr += vl; dw_ptr += vl; rem -= vl;
    }

    successful_steps++;
    if (successful_steps > 200) {
        dynamic_loss_scale *= (_Float16)2.0f;
        successful_steps    = 0;
    }
    return 1;
}

void fc_op_backward_input_only(fc_op *op)
{
    printf_("[SCALAR] fc_op_backward_input_only: batchsize=%d in=%d out=%d\n",
            op->batchsize, op->in_units, op->out_units);
    for (int p = 0; p < op->batchsize; p++)
    {
        for (int j = 0; j < op->out_units; j++)
        {
            register _Float16 d_o = op->d_output[p * op->out_units + j];
            for (int i = 0; i < op->in_units; i++)
                op->d_input[p * op->in_units + i] += op->weights[i * op->out_units + j] * d_o;
        }
    }
}

static void vzero_f16(_Float16 *buf, size_t n)
{
    _Float16 *ptr = buf;
    while (n > 0) {
        size_t vl;
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(n));
        asm volatile("vmv.v.i v8, 0");
        asm volatile("vse16.v v8, (%0)" :: "r"(ptr) : "memory");
        ptr += vl; n -= vl;
    }
}

void calloc_fc_weights(fc_op *op)
{
    if (op->weights)
        vzero_f16(op->weights, (size_t)op->in_units * op->out_units);
    if (op->bias)
        vzero_f16(op->bias, (size_t)op->out_units);
}

void free_fc_weights(fc_op *op)
{
    (void)op;
}

void calloc_fc_dweights(fc_op *op)
{
    if (op->d_weights)
        vzero_f16(op->d_weights, (size_t)op->in_units * op->out_units);
    if (op->d_bias)
        vzero_f16(op->d_bias, (size_t)op->out_units);
}

void free_fc_dweights(fc_op *op)
{
    (void)op;
}

void save_fc_weights(fc_op *op)
{
    (void)op;
}

void load_fc_weights(fc_op *op, _Float16 *w_array, _Float16 *b_array)
{
    // Pre-absorb INPUT_SCALE into the weights so the fast-path forward pass
    // (no padding) can use the input buffer directly without a separate scale step.
    // Equivalent: (A * INPUT_SCALE) * W  ==  A * (W * INPUT_SCALE)
    const size_t nw = (size_t)op->in_units * (size_t)op->out_units;
    size_t rem = nw;
    const _Float16 *src = w_array;
    _Float16 *dst = op->weights;
    while (rem > 0)
    {
        size_t vl = 0;
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(rem));
        asm volatile("vle16.v v0, (%0)"     : : "r"(src) : "memory");
        asm volatile("vfmul.vf v0, v0, %0"  : : "f"(INPUT_SCALE));
        asm volatile("vse16.v v0, (%0)"     : : "r"(dst) : "memory");
        src += vl; dst += vl; rem -= vl;
    }
    memcpy(op->bias, b_array, sizeof(_Float16) * op->out_units);
}

static void matrix_multiply_scalar_fused(const _Float16 *a, const _Float16 *b,
                                         const _Float16 *bias, _Float16 *c,
                                         const int M, const int N, const int K)
{
    register int i, j, p;
    register const _Float16 *a_ptr = a;
    for (i = 0; i < M; i++)
    {
        _Float16 *c_ptr = c + i * K;
        const _Float16 *bias_ptr = bias;
        for (p = 0; p < K; p++)
            *(c_ptr++) = *(bias_ptr++);
    }
    for (i = 0; i < M; i++)
    {
        register const _Float16 *b_ptr = b;
        for (j = 0; j < N; j++)
        {
            register _Float16 apart = *(a_ptr++);
            if (apart == (_Float16)0.0f)
            {
                b_ptr += K;
                continue;
            }
            register _Float16 *c_ptr = c + i * K;
            for (p = 0; p < K; p++)
                *(c_ptr++) += *(b_ptr++) * apart;
        }
    }
}

void calc_bias_gradient_vec_16(_Float16 *d_bias, const _Float16 *d_output,
                                int out_units, int batchsize)
{
    _Float16 inv_batch = (_Float16)(1.0f / (float)batchsize);

    for (int p = 0; p < batchsize; p++) {
        const _Float16 *row = d_output + (size_t)p * out_units;
        _Float16 *bias_ptr = d_bias;
        int j = out_units;
        while (j > 0) {
            size_t vl;
            asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(j));
            asm volatile("vle16.v v8,  (%0)" :: "r"(bias_ptr));
            asm volatile("vle16.v v16, (%0)" :: "r"(row));
            asm volatile("vfadd.vv v8, v8, v16");
            asm volatile("vse16.v v8,  (%0)" :: "r"(bias_ptr));
            bias_ptr += vl;
            row      += vl;
            j        -= (int)vl;
        }
    }

    _Float16 *ptr = d_bias;
    int n = out_units;
    while (n > 0) {
        size_t vl;
        asm volatile("vsetvli %0, %1, e16, m8, ta, ma" : "=r"(vl) : "r"(n));
        asm volatile("vle16.v v8, (%0)"     :: "r"(ptr));
        asm volatile("vfmul.vf v8, v8, %0" :: "f"(inv_batch));
        asm volatile("vse16.v v8, (%0)"     :: "r"(ptr));
        ptr += vl;
        n   -= (int)vl;
    }
}
