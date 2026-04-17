//
// File:        matrix.c
// Description: Implementation of matrix computation
// Author:      Haris Wang
//
#include <stdlib.h>
#include <string.h>
#include "alexnet.h"
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
    register int i,j,p;
    register float *a_ptr = a;
    for (i = 0; i < M; i++)
    {
        register float *b_ptr = b;
        for (j = 0; j < N; j++)
        {
            register float apart = *(a_ptr++);
            if (apart<0.00001 && apart>(0-0.00001)) //masking for vector processing
            {
                b_ptr += K;
                continue;
            }
            register float *c_ptr = c + i*K; //reset
            for (p = 0; p < K; p++)
                *(c_ptr++) += *(b_ptr++) * apart;
        }
    }
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
