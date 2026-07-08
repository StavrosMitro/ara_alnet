// Copyright 2020 ETH Zurich and University of Bologna.
//
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Author: Matheus Cavalcante, ETH Zurich
//         Samuel Riedel, ETH Zurich
// Pure FP16 refactor: Stavros Mitropoulos, NTUA

#include <stddef.h>
#include <stdint.h>
#include "fmatmul.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))

// ---------------------------------------------------------------------------
// Dispatch: select tile size by M (NT version - C = A * B^T)
// ---------------------------------------------------------------------------
void fmatmul_nt_16(_Float16 *c, const _Float16 *a, const _Float16 *b,
                   const unsigned long int M, const unsigned long int N,
                   const unsigned long int P) {
    if (M <= 4) {
        fmatmul_4x4_nt_16(c, a, b, M, N, P);
    } else if (M <= 8) {
        fmatmul_8x8_nt_16(c, a, b, M, N, P);
    } else if (M <= 64) {
        fmatmul_16x16_nt_16(c, a, b, M, N, P);
    } else if (M <= 128) {
        fmatmul_8x8_nt_16(c, a, b, M, N, P);
    } else {
        fmatmul_4x4_nt_16(c, a, b, M, N, P);
    }
}

// ---------------
// 4x4 NT
// ---------------

void fmatmul_4x4_nt_16(_Float16 *c, const _Float16 *a, const _Float16 *b,
                        const unsigned long int M, const unsigned long int N,
                        const unsigned long int P) {
    const unsigned long int block_size = 4;

    for (unsigned long int p = 0; p < P; p += block_size) {
        const unsigned long int p_ = MIN(P - p, block_size);

        const _Float16 *b_ = b + p * N;
        _Float16 *c_ = c + p;

        unsigned long int block_size_n;
        asm volatile("vsetvli %0, %1, e16, m4, ta, ma" : "=r"(block_size_n) : "r"(N));

        for (unsigned long int m = 0; m < M; m += block_size) {
            const _Float16 *a_ = a + m * N;
            _Float16 *c__ = c_ + m * P;

            fmatmul_vec_4x4_slice_init_nt_16();
            fmatmul_vec_4x4_nt_16(c__, a_, b_, N, P, p_);
        }
    }
}

void fmatmul_vec_4x4_slice_init_nt_16(void) {
    asm volatile("vmv.v.i v0,  0");
    asm volatile("vmv.v.i v4,  0");
    asm volatile("vmv.v.i v8,  0");
    asm volatile("vmv.v.i v12, 0");
}

void fmatmul_vec_4x4_nt_16(_Float16 *c, const _Float16 *a, const _Float16 *b,
                            const unsigned long int N, const unsigned long int P,
                            const unsigned long int p_) {
    unsigned long int block_size_n;
    asm volatile("vsetvli %0, %1, e16, m4, ta, ma" : "=r"(block_size_n) : "r"(N));

    for (unsigned long int n = 0; n < N; n += block_size_n) {
        const unsigned long int n_ = MIN(N - n, block_size_n);

        asm volatile("vsetvli zero, %0, e16, m4, ta, ma" :: "r"(n_));

        asm volatile("vle16.v v16, (%0);" :: "r"(a + n));
        asm volatile("vle16.v v20, (%0);" :: "r"(b + n));

        asm volatile("vfmacc.vv v0,  v16, v20");
        asm volatile("vfmacc.vv v4,  v16, v20");
        asm volatile("vfmacc.vv v8,  v16, v20");
        asm volatile("vfmacc.vv v12, v16, v20");
    }

    asm volatile("vsetvli zero, %0, e16, m4, ta, ma" :: "r"(p_));
    asm volatile("vfredusum.vs v0, v0, v0");
    asm volatile("vfredusum.vs v4, v4, v4");
    asm volatile("vfredusum.vs v8, v8, v8");
    asm volatile("vfredusum.vs v12, v12, v12");

    _Float16 res0, res1, res2, res3;
    asm volatile("vfmv.f.s %0, v0" : "=f"(res0));
    asm volatile("vfmv.f.s %0, v4" : "=f"(res1));
    asm volatile("vfmv.f.s %0, v8" : "=f"(res2));
    asm volatile("vfmv.f.s %0, v12" : "=f"(res3));

    *c = res0; c += P;
    *c = res1; c += P;
    *c = res2; c += P;
    *c = res3;
}

// ---------------
// 8x8 NT
// ---------------

void fmatmul_8x8_nt_16(_Float16 *c, const _Float16 *a, const _Float16 *b,
                        const unsigned long int M, const unsigned long int N,
                        const unsigned long int P) {
    const unsigned long int block_size = 8;

    for (unsigned long int p = 0; p < P; p += block_size) {
        const unsigned long int p_ = MIN(P - p, block_size);

        const _Float16 *b_ = b + p * N;
        _Float16 *c_ = c + p;

        unsigned long int block_size_n;
        asm volatile("vsetvli %0, %1, e16, m2, ta, ma" : "=r"(block_size_n) : "r"(N));

        for (unsigned long int m = 0; m < M; m += block_size) {
            const _Float16 *a_ = a + m * N;
            _Float16 *c__ = c_ + m * P;

            fmatmul_vec_8x8_slice_init_nt_16();
            fmatmul_vec_8x8_nt_16(c__, a_, b_, N, P, p_);
        }
    }
}

void fmatmul_vec_8x8_slice_init_nt_16(void) {
    asm volatile("vmv.v.i v0,  0");
    asm volatile("vmv.v.i v2,  0");
    asm volatile("vmv.v.i v4,  0");
    asm volatile("vmv.v.i v6,  0");
    asm volatile("vmv.v.i v8,  0");
    asm volatile("vmv.v.i v10, 0");
    asm volatile("vmv.v.i v12, 0");
    asm volatile("vmv.v.i v14, 0");
}

void fmatmul_vec_8x8_nt_16(_Float16 *c, const _Float16 *a, const _Float16 *b,
                            const unsigned long int N, const unsigned long int P,
                            const unsigned long int p_) {
    unsigned long int block_size_n;
    asm volatile("vsetvli %0, %1, e16, m2, ta, ma" : "=r"(block_size_n) : "r"(N));

    for (unsigned long int n = 0; n < N; n += block_size_n) {
        const unsigned long int n_ = MIN(N - n, block_size_n);

        asm volatile("vsetvli zero, %0, e16, m2, ta, ma" :: "r"(n_));

        asm volatile("vle16.v v16, (%0);" :: "r"(a + n));
        asm volatile("vle16.v v18, (%0);" :: "r"(b + n));

        asm volatile("vfmacc.vv v0,  v16, v18");
        asm volatile("vfmacc.vv v2,  v16, v18");
        asm volatile("vfmacc.vv v4,  v16, v18");
        asm volatile("vfmacc.vv v6,  v16, v18");
        asm volatile("vfmacc.vv v8,  v16, v18");
        asm volatile("vfmacc.vv v10, v16, v18");
        asm volatile("vfmacc.vv v12, v16, v18");
        asm volatile("vfmacc.vv v14, v16, v18");
    }

    asm volatile("vsetvli zero, %0, e16, m2, ta, ma" :: "r"(p_));
    asm volatile("vfredusum.vs v0, v0, v0");
    asm volatile("vfredusum.vs v2, v2, v2");
    asm volatile("vfredusum.vs v4, v4, v4");
    asm volatile("vfredusum.vs v6, v6, v6");
    asm volatile("vfredusum.vs v8, v8, v8");
    asm volatile("vfredusum.vs v10, v10, v10");
    asm volatile("vfredusum.vs v12, v12, v12");
    asm volatile("vfredusum.vs v14, v14, v14");

    _Float16 res0, res1, res2, res3, res4, res5, res6, res7;
    asm volatile("vfmv.f.s %0, v0" : "=f"(res0));
    asm volatile("vfmv.f.s %0, v2" : "=f"(res1));
    asm volatile("vfmv.f.s %0, v4" : "=f"(res2));
    asm volatile("vfmv.f.s %0, v6" : "=f"(res3));
    asm volatile("vfmv.f.s %0, v8" : "=f"(res4));
    asm volatile("vfmv.f.s %0, v10" : "=f"(res5));
    asm volatile("vfmv.f.s %0, v12" : "=f"(res6));
    asm volatile("vfmv.f.s %0, v14" : "=f"(res7));

    *c = res0; c += P;
    *c = res1; c += P;
    *c = res2; c += P;
    *c = res3; c += P;
    *c = res4; c += P;
    *c = res5; c += P;
    *c = res6; c += P;
    *c = res7;
}

// ---------------
// 16x16 NT
// ---------------

void fmatmul_16x16_nt_16(_Float16 *c, const _Float16 *a, const _Float16 *b,
                         unsigned long int M, unsigned long int N, unsigned long int P) {
    const unsigned long int block_size = 16;

    for (unsigned long int p = 0; p < P; p += block_size) {
        const unsigned long int p_ = MIN(P - p, block_size);

        const _Float16 *b_ = b + p * N;
        _Float16 *c_ = c + p;

        unsigned long int block_size_n;
        asm volatile("vsetvli %0, %1, e16, m1, ta, ma" : "=r"(block_size_n) : "r"(N));

        for (unsigned long int m = 0; m < M; m += block_size) {
            const _Float16 *a_ = a + m * N;
            _Float16 *c__ = c_ + m * P;

            fmatmul_vec_16x16_slice_init_nt_16();
            fmatmul_vec_16x16_nt_16(c__, a_, b_, N, P, p_);
        }
    }
}

void fmatmul_vec_16x16_slice_init_nt_16(void) {
    asm volatile("vmv.v.i v0,  0"); asm volatile("vmv.v.i v1,  0");
    asm volatile("vmv.v.i v2,  0"); asm volatile("vmv.v.i v3,  0");
    asm volatile("vmv.v.i v4,  0"); asm volatile("vmv.v.i v5,  0");
    asm volatile("vmv.v.i v6,  0"); asm volatile("vmv.v.i v7,  0");
    asm volatile("vmv.v.i v8,  0"); asm volatile("vmv.v.i v9,  0");
    asm volatile("vmv.v.i v10, 0"); asm volatile("vmv.v.i v11, 0");
    asm volatile("vmv.v.i v12, 0"); asm volatile("vmv.v.i v13, 0");
    asm volatile("vmv.v.i v14, 0"); asm volatile("vmv.v.i v15, 0");
}

void fmatmul_vec_16x16_nt_16(_Float16 *c, const _Float16 *a, const _Float16 *b,
                              const unsigned long int N, const unsigned long int P,
                              const unsigned long int p_) {
    unsigned long int block_size_n;
    asm volatile("vsetvli %0, %1, e16, m1, ta, ma" : "=r"(block_size_n) : "r"(N));

    for (unsigned long int n = 0; n < N; n += block_size_n) {
        const unsigned long int n_ = MIN(N - n, block_size_n);

        asm volatile("vsetvli zero, %0, e16, m1, ta, ma" :: "r"(n_));

        asm volatile("vle16.v v16, (%0);" :: "r"(a + n));
        asm volatile("vle16.v v17, (%0);" :: "r"(b + n));

        asm volatile("vfmacc.vv v0,  v16, v17");
        asm volatile("vfmacc.vv v1,  v16, v17");
        asm volatile("vfmacc.vv v2,  v16, v17");
        asm volatile("vfmacc.vv v3,  v16, v17");
        asm volatile("vfmacc.vv v4,  v16, v17");
        asm volatile("vfmacc.vv v5,  v16, v17");
        asm volatile("vfmacc.vv v6,  v16, v17");
        asm volatile("vfmacc.vv v7,  v16, v17");
        asm volatile("vfmacc.vv v8,  v16, v17");
        asm volatile("vfmacc.vv v9,  v16, v17");
        asm volatile("vfmacc.vv v10, v16, v17");
        asm volatile("vfmacc.vv v11, v16, v17");
        asm volatile("vfmacc.vv v12, v16, v17");
        asm volatile("vfmacc.vv v13, v16, v17");
        asm volatile("vfmacc.vv v14, v16, v17");
        asm volatile("vfmacc.vv v15, v16, v17");
    }

    asm volatile("vsetvli zero, %0, e16, m1, ta, ma" :: "r"(p_));
    asm volatile("vfredusum.vs v0, v0, v0");
    asm volatile("vfredusum.vs v1, v1, v1");
    asm volatile("vfredusum.vs v2, v2, v2");
    asm volatile("vfredusum.vs v3, v3, v3");
    asm volatile("vfredusum.vs v4, v4, v4");
    asm volatile("vfredusum.vs v5, v5, v5");
    asm volatile("vfredusum.vs v6, v6, v6");
    asm volatile("vfredusum.vs v7, v7, v7");
    asm volatile("vfredusum.vs v8, v8, v8");
    asm volatile("vfredusum.vs v9, v9, v9");
    asm volatile("vfredusum.vs v10, v10, v10");
    asm volatile("vfredusum.vs v11, v11, v11");
    asm volatile("vfredusum.vs v12, v12, v12");
    asm volatile("vfredusum.vs v13, v13, v13");
    asm volatile("vfredusum.vs v14, v14, v14");
    asm volatile("vfredusum.vs v15, v15, v15");

    _Float16 res[16];
    asm volatile("vfmv.f.s %0, v0" : "=f"(res[0]));
    asm volatile("vfmv.f.s %0, v1" : "=f"(res[1]));
    asm volatile("vfmv.f.s %0, v2" : "=f"(res[2]));
    asm volatile("vfmv.f.s %0, v3" : "=f"(res[3]));
    asm volatile("vfmv.f.s %0, v4" : "=f"(res[4]));
    asm volatile("vfmv.f.s %0, v5" : "=f"(res[5]));
    asm volatile("vfmv.f.s %0, v6" : "=f"(res[6]));
    asm volatile("vfmv.f.s %0, v7" : "=f"(res[7]));
    asm volatile("vfmv.f.s %0, v8" : "=f"(res[8]));
    asm volatile("vfmv.f.s %0, v9" : "=f"(res[9]));
    asm volatile("vfmv.f.s %0, v10" : "=f"(res[10]));
    asm volatile("vfmv.f.s %0, v11" : "=f"(res[11]));
    asm volatile("vfmv.f.s %0, v12" : "=f"(res[12]));
    asm volatile("vfmv.f.s %0, v13" : "=f"(res[13]));
    asm volatile("vfmv.f.s %0, v14" : "=f"(res[14]));
    asm volatile("vfmv.f.s %0, v15" : "=f"(res[15]));

    for (int i = 0; i < 16; i++) {
        *c = res[i]; c += P;
    }
}
