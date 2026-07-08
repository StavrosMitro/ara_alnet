// Copyright 2020 ETH Zurich and University of Bologna.
//
// SPDX-License-Identifier: Apache-2.0
//
// Author: Matteo Perotti (original 7x7), adapted for 3x3 float
// Mixed precision (FP16 input/weights → FP32 accumulator): Stavros Mitropoulos

#include "fconv3d.h"
#include <stdint.h>

// ============================================================================
// fconv3d_CHx3x3_f16 — mixed-precision 3×3 convolution
//
// Input tensor  (i): FP16  (_Float16*)
// Filter tensor (f): FP16  (_Float16*)  — packed per output-channel by caller
// Output tensor (o): FP32  (float*)     — widening accumulator
// Bias          : FP32  (float)
//
// LMUL strategy:
//   e16, m1  — source SEW/LMUL for loads and slides
//   e32, m2  — destination SEW/LMUL for the widening accumulator v20
//   VLMAX(e16,m1) == VLMAX(e32,m2), so vl is valid for both configs.
//
// Widening FMA: vfwmacc.vf v20, fc_f16, vi
//   reads vi as e16,m1 (FP16) and fc_f16 as FP16 scalar (low 16 bits of freg)
//   widens product to FP32 and accumulates into v20 (e32,m2).
// ============================================================================

static void fconv3d_CHx3x3_block_f16(float *o, const _Float16 *i, const _Float16 *f,
                                      int64_t M, int64_t N, int64_t n_,
                                      int64_t C, float bias)
{
    int64_t ldo      = N << 2;           // output row stride: N × sizeof(float)
    int64_t ldi_pad  = (N + 2) << 1;    // input  row stride: (N+2) × sizeof(_Float16)
    int64_t ich_len  = (M + 2) * (N + 2);
    int64_t fch_len  = 9;

    const _Float16 *i_row = i;
    float          *o_row = o;

    for (int64_t m = 0; m < M; ++m) {
        // ---- init FP32 accumulator v20 (e32,m2) with bias ----
        // vsetvli zero, zero retains current vl while changing vtype.
        // VLMAX(e32,m2) == VLMAX(e16,m1), so the existing vl stays valid.
        asm volatile("vsetvli zero, zero, e32, m2, ta, ma");
        asm volatile("vfmv.v.f v20, %0" :: "f"(bias));

        // ---- switch to FP16 source config for the channel loop ----
        asm volatile("vsetvli zero, zero, e16, m1, ta, ma");

        for (int64_t ch = 0; ch < C; ++ch) {
            const _Float16 *row0 = i_row + ch * ich_len;
            const _Float16 *row1 = row0 + (N + 2);
            const _Float16 *row2 = row1 + (N + 2);

            const _Float16 *slide0 = row0 + n_;
            const _Float16 *slide1 = row1 + n_;
            const _Float16 *slide2 = row2 + n_;

            // Load n_ FP16 pixels for each of the 3 input rows
            asm volatile("vle16.v v0,  (%0)" :: "r"(row0));
            asm volatile("vle16.v v6,  (%0)" :: "r"(row1));
            asm volatile("vle16.v v12, (%0)" :: "r"(row2));

            // Two slide-right operations per row expose all 3 horizontal kernel positions
            asm volatile("vfslide1down.vf v2,  v0,  %0" :: "f"(*slide0++));
            asm volatile("vfslide1down.vf v4,  v2,  %0" :: "f"(*slide0++));
            asm volatile("vfslide1down.vf v8,  v6,  %0" :: "f"(*slide1++));
            asm volatile("vfslide1down.vf v10, v8,  %0" :: "f"(*slide1++));
            asm volatile("vfslide1down.vf v14, v12, %0" :: "f"(*slide2++));
            asm volatile("vfslide1down.vf v16, v14, %0" :: "f"(*slide2++));

            // 9-tap widening FMA: FP16×FP16 → FP32 accumulate into v20 (e32,m2)
            const _Float16 *fc = f + ch * fch_len;
            asm volatile("vfwmacc.vf v20, %0, v0"  :: "f"(fc[0]));
            asm volatile("vfwmacc.vf v20, %0, v2"  :: "f"(fc[1]));
            asm volatile("vfwmacc.vf v20, %0, v4"  :: "f"(fc[2]));
            asm volatile("vfwmacc.vf v20, %0, v6"  :: "f"(fc[3]));
            asm volatile("vfwmacc.vf v20, %0, v8"  :: "f"(fc[4]));
            asm volatile("vfwmacc.vf v20, %0, v10" :: "f"(fc[5]));
            asm volatile("vfwmacc.vf v20, %0, v12" :: "f"(fc[6]));
            asm volatile("vfwmacc.vf v20, %0, v14" :: "f"(fc[7]));
            asm volatile("vfwmacc.vf v20, %0, v16" :: "f"(fc[8]));
        }

        // ---- store FP32 accumulator ----
        asm volatile("vsetvli zero, zero, e32, m2, ta, ma");
        asm volatile("vse32.v v20, (%0)" :: "r"(o_row));

        o_row = (float          *)((uintptr_t)o_row + ldo);
        i_row = (const _Float16 *)((uintptr_t)i_row + ldi_pad);
    }
}

void fconv3d_CHx3x3_f16(float *o, const _Float16 *i, const _Float16 *f, int64_t M,
                         int64_t N, int64_t C, float bias)
{
    unsigned long int block_size_n;
    asm volatile("vsetvli %0, %1, e16, m1, ta, ma" : "=r"(block_size_n) : "r"(N));

    for (unsigned long int n = 0; n < (unsigned long int)N; n += block_size_n) {
        const unsigned long int n_ = MIN((unsigned long int)N - n, block_size_n);
        asm volatile("vsetvli zero, %0, e16, m1, ta, ma" :: "r"(n_));
        fconv3d_CHx3x3_block_f16(o + n, i + n, f, M, N, (int64_t)n_, C, bias);
    }
}

// ============================================================================
// fconv3d_CHx3x3_f16_f16out — same widening FMA kernel, but stores FP16 output
//
// After the FP32 accumulation finishes for each row, vfncvt.f.f.w narrows
// v20 (e32,m2) → v0 (e16,m1) and stores FP16 into the output array.
// Output stride: N × sizeof(_Float16) = N × 2 bytes.
// ============================================================================
static void fconv3d_CHx3x3_block_f16_f16out(_Float16 *o, const _Float16 *i, const _Float16 *f,
                                             int64_t M, int64_t N, int64_t n_,
                                             int64_t C, float bias)
{
    int64_t ldo      = N << 1;            // output row stride: N × sizeof(_Float16)
    int64_t ldi_pad  = (N + 2) << 1;     // input  row stride: (N+2) × sizeof(_Float16)
    int64_t ich_len  = (M + 2) * (N + 2);
    int64_t fch_len  = 9;

    const _Float16 *i_row = i;
    _Float16       *o_row = o;

    for (int64_t m = 0; m < M; ++m) {
        asm volatile("vsetvli zero, zero, e32, m2, ta, ma");
        asm volatile("vfmv.v.f v20, %0" :: "f"(bias));
        asm volatile("vsetvli zero, zero, e16, m1, ta, ma");

        for (int64_t ch = 0; ch < C; ++ch) {
            const _Float16 *row0 = i_row + ch * ich_len;
            const _Float16 *row1 = row0 + (N + 2);
            const _Float16 *row2 = row1 + (N + 2);

            const _Float16 *slide0 = row0 + n_;
            const _Float16 *slide1 = row1 + n_;
            const _Float16 *slide2 = row2 + n_;

            asm volatile("vle16.v v0,  (%0)" :: "r"(row0));
            asm volatile("vle16.v v6,  (%0)" :: "r"(row1));
            asm volatile("vle16.v v12, (%0)" :: "r"(row2));

            asm volatile("vfslide1down.vf v2,  v0,  %0" :: "f"(*slide0++));
            asm volatile("vfslide1down.vf v4,  v2,  %0" :: "f"(*slide0++));
            asm volatile("vfslide1down.vf v8,  v6,  %0" :: "f"(*slide1++));
            asm volatile("vfslide1down.vf v10, v8,  %0" :: "f"(*slide1++));
            asm volatile("vfslide1down.vf v14, v12, %0" :: "f"(*slide2++));
            asm volatile("vfslide1down.vf v16, v14, %0" :: "f"(*slide2++));

            const _Float16 *fc = f + ch * fch_len;
            asm volatile("vfwmacc.vf v20, %0, v0"  :: "f"(fc[0]));
            asm volatile("vfwmacc.vf v20, %0, v2"  :: "f"(fc[1]));
            asm volatile("vfwmacc.vf v20, %0, v4"  :: "f"(fc[2]));
            asm volatile("vfwmacc.vf v20, %0, v6"  :: "f"(fc[3]));
            asm volatile("vfwmacc.vf v20, %0, v8"  :: "f"(fc[4]));
            asm volatile("vfwmacc.vf v20, %0, v10" :: "f"(fc[5]));
            asm volatile("vfwmacc.vf v20, %0, v12" :: "f"(fc[6]));
            asm volatile("vfwmacc.vf v20, %0, v14" :: "f"(fc[7]));
            asm volatile("vfwmacc.vf v20, %0, v16" :: "f"(fc[8]));
        }

        // Narrow FP32 accumulator → FP16 output
        // VLMAX(e32,m2) == VLMAX(e16,m1), so vl is valid for both configs.
        asm volatile("vsetvli zero, zero, e16, m1, ta, ma");
        asm volatile("vfncvt.f.f.w v0, v20");
        asm volatile("vse16.v v0, (%0)" :: "r"(o_row));

        o_row = (_Float16 *)((uintptr_t)o_row + ldo);
        i_row = (const _Float16 *)((uintptr_t)i_row + ldi_pad);
    }
}

void fconv3d_CHx3x3_f16_f16out(_Float16 *o, const _Float16 *i, const _Float16 *f, int64_t M,
                                int64_t N, int64_t C, float bias)
{
    unsigned long int block_size_n;
    asm volatile("vsetvli %0, %1, e16, m1, ta, ma" : "=r"(block_size_n) : "r"(N));

    for (unsigned long int n = 0; n < (unsigned long int)N; n += block_size_n) {
        const unsigned long int n_ = MIN((unsigned long int)N - n, block_size_n);
        asm volatile("vsetvli zero, %0, e16, m1, ta, ma" :: "r"(n_));
        fconv3d_CHx3x3_block_f16_f16out(o + n, i + n, f, M, N, (int64_t)n_, C, bias);
    }
}
