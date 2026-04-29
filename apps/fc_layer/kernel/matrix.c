//
// File:        matrix.c
// Description: Implementation of matrix computation
// Author:      Haris Wang
// Modified: Stavros Mitropoulos
#include <stdlib.h>
#include <string.h>
#include "alexnet.h"
#include "fmatmul.h"
// #include <immintrin.h> 
#ifdef SPIKE
#include <printf.h>
#elif defined ARA_LINUX
#include <stdio.h>
#else
#include "printf.h"
#endif

// I REMOVED IT, BECAUSE IT IS FOR X86 ASSEMBLY

// Workspace used by matrix_transpose to avoid large stack allocations.
// Sized conservatively for current AlexNet transpose use cases.
#define MATRIX_TRANSPOSE_WORKSPACE_ELEMS 2000000
static float matrix_transpose_workspace[MATRIX_TRANSPOSE_WORKSPACE_ELEMS];

#if ALEXNET_STATIC_MAX_BATCH > 4
#define FMATMUL_MAX_M ALEXNET_STATIC_MAX_BATCH
#else
#define FMATMUL_MAX_M 4
#endif

#define FMATMUL_MAX_N FC_MAX_IN_UNITS
#define FMATMUL_MAX_K FC_MAX_INTERNAL

static float fmatmul_a_scratch[FMATMUL_MAX_M * FMATMUL_MAX_N];
static float fmatmul_b_scratch[FMATMUL_MAX_N * FMATMUL_MAX_K];
static float fmatmul_c_scratch[FMATMUL_MAX_M * FMATMUL_MAX_K];

static inline unsigned long int fmatmul_row_block(unsigned long int m)
{
    if (m <= 4)
        return 4;
    if (m <= 8)
        return 8;
    if (m <= 64)
        return 16;
    if (m <= 128)
        return 8;
    return 4;
}

static void matrix_multiply_scalar(const float *a, const float *b, float *c,
                                   const int M, const int N, const int K)
{
    register int i, j, p;
    register const float *a_ptr = a;
    for (i = 0; i < M; i++)
    {
        register const float *b_ptr = b;
        for (j = 0; j < N; j++)
        {
            register float apart = *(a_ptr++);
            if (apart < 0.00001f && apart > -0.00001f)
            {
                b_ptr += K;
                continue;
            }
            register float *c_ptr = c + i * K;
            for (p = 0; p < K; p++)
                *(c_ptr++) += *(b_ptr++) * apart;
        }
    }
}

void matrix_multiply(const float *a, const float *b, float *c, const int M, const int N, const int K)
{
    /**
     * matrix multiply, c = a * b
     * 
     * Input:
     * a    [M,N]
     * b    [N,K]
     * Output:
     * c    [M,K]
     * */
    if (M <= 0 || N <= 0 || K <= 0)
        return;

    unsigned long int block = fmatmul_row_block((unsigned long int)M);
    unsigned long int padded_m = (((unsigned long int)M + block - 1) / block) * block;

    if ((unsigned long int)N > FMATMUL_MAX_N ||
        (unsigned long int)K > FMATMUL_MAX_K ||
        padded_m > FMATMUL_MAX_M)
    {
        matrix_multiply_scalar(a, b, c, M, N, K);
        return;
    }

    const size_t mn = (size_t)M * (size_t)N;
    const size_t nk = (size_t)N * (size_t)K;
    const size_t pnk = (size_t)padded_m * (size_t)N;
    const size_t mk = (size_t)M * (size_t)K;

    for (size_t idx = 0; idx < mn; idx++) //stripmining
        fmatmul_a_scratch[idx] = a[idx];
    for (size_t idx = mn; idx < pnk; idx++)
        fmatmul_a_scratch[idx] = 0.0f;

    fmatmul(fmatmul_c_scratch, fmatmul_a_scratch, b,
            padded_m, (unsigned long int)N, (unsigned long int)K);

    for (size_t idx = 0; idx < mk; idx++)
        c[idx] += fmatmul_c_scratch[idx];
}

void matrix_transpose(float *x, int m, int n)
{
    /** matrix transpose
     * 
     * Input:
     *      x[m,n]
     * Output:
     *      x[n,m]
     * */
    size_t elems = (size_t)m * (size_t)n;
    if (elems > MATRIX_TRANSPOSE_WORKSPACE_ELEMS) {
        printf_("Error: matrix_transpose workspace too small for %d x %d\n", m, n);
        exit(1);
    }
    float *tmp = matrix_transpose_workspace;
    register int i, j;
    register float *ptr = x;
    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n; j++)
            tmp[j*m+i] = *(ptr++);
    }
    memcpy(x, tmp, elems * sizeof(float));
    return;
}
