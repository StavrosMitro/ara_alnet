//
// File:        fc_layer.c
// Description: Implementation of full connected layer
// Author:      Haris Wang
// Modified: Stavros Mitropoulos
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
#define MIN(a,b) (((a) < (b)) ? (a) : (b))

static inline int64_t fc_cycle_count_local(void)
{
    int64_t cycle_count = 0;
    asm volatile("fence; csrr %0, cycle" : "=r"(cycle_count));
    return cycle_count;
}

static float fc_t_weights_scratch[FC_MAX_IN_UNITS * FC_MAX_INTERNAL];
static float fc_t_output_scratch[ALEXNET_STATIC_MAX_BATCH * FC_MAX_INTERNAL];


typedef struct fc_args{
    fc_op *op;
    short batch_id;
    short st_tunits;
    short ed_tunits;
} fc_args;

static void fc_op_forward_range(void *argv)
{
    /**
     * pthread fc_op_forward
     * */
    fc_args args;
    memcpy(&args, (fc_args *)argv, sizeof(fc_args));
    short internal   = args.ed_tunits - args.st_tunits;
    if (internal > FC_MAX_INTERNAL || args.op->in_units > FC_MAX_IN_UNITS || args.op->batchsize > ALEXNET_STATIC_MAX_BATCH) {
        printf_("Error: FC scratch bounds exceeded (internal=%d, in_units=%d, batch=%d)\n",
               (int)internal, args.op->in_units, args.op->batchsize);
        exit(1);
    }
    float *t_weights = fc_t_weights_scratch;
    
    for (int j = 0; j < args.op->in_units; j++)
    {   
        memcpy((void *)(t_weights+j*internal), 
                (void *)(args.op->weights+j*(args.op->out_units)+args.st_tunits), 
                    sizeof(float)*internal);
    }

    float *t_output = fc_t_output_scratch;
    memset(t_output, 0, (size_t)internal * args.op->batchsize * sizeof(float));
    matrix_multiply(args.op->input, t_weights, t_output,  args.op->batchsize,  args.op->in_units, internal);

    for (int j = 0; j < args.op->batchsize; j++)
    {
        register int o_offset  = j * internal;
        register int oo_offset = j * (args.op->out_units) + args.st_tunits;

        for (int i = 0; i < internal; i++, o_offset++, oo_offset++)
            args.op->output[oo_offset] = t_output[o_offset] + args.op->bias[args.st_tunits+i]; 
    }
    return;
}

void fc_op_forward(fc_op *op)
{
    short tnum = 8; // number of partitions
    if (op->out_units < tnum)
    {
        fc_args args;
        args.op = op;
        args.st_tunits = 0;
        args.ed_tunits = op->out_units;
        fc_op_forward_range((void *)(&args));
    }else {
        fc_args args[tnum+1];
        short internal = ceil(1.0 * op->out_units / tnum);
    
        for (int p = 0; p < tnum; p++)
        {
            args[p].op = op;
            args[p].st_tunits = p*internal;
            args[p].ed_tunits = MIN(args[p].st_tunits+internal, op->out_units);
            if (args[p].st_tunits >= op->out_units) break;
            fc_op_forward_range((void *)(&args[p]));
        }
    }

}

static void fc_op_backward_range(void *argv)
{
    /**
     * pthread fc_op_backward
     * */
    fc_args args;
    memcpy(&args, (fc_args *)argv, sizeof(fc_args));

    if (args.st_tunits == 0)
    {
        // calculate delta_input per sample
        // weights layout: W[in_units, out_units], W[i][j] = weights[i * out_units + j]
        for (int p = 0; p < args.op->batchsize; p++)
        {
            for (register int j = 0; j < args.op->out_units; j++)
            {
                register float d_o = args.op->d_output[p * args.op->out_units + j];
                for (register int i = 0; i < args.op->in_units; i++)
                {
                    args.op->d_input[p * args.op->in_units + i] += args.op->weights[i * args.op->out_units + j] * d_o;
                }
            }
        }
        // calculate delta_bias averaged across batch
        for (register int j = 0; j < args.op->out_units; j++)
        {
            register float sum = 0.0f;
            for (int p = 0; p < args.op->batchsize; p++)
                sum += args.op->d_output[p * args.op->out_units + j];
            args.op->d_bias[j] = sum / args.op->batchsize;
        }
    }
 
    register float *w_deltas  = args.op->d_weights;
    for (int i = args.st_tunits; i < args.ed_tunits; i++)
    {
        register float *input = args.op->input;
        // d_weights layout must match weights: W[k][i] = d_weights[k * out_units + i]
        for (int p = 0; p < args.op->batchsize; p++)
        {
            register float oe = args.op->d_output[p * args.op->out_units + i];
            if (fabsf(oe) < 1e-9f)
                continue;

            for (int k = 0; k < args.op->in_units; k++)
            {
                w_deltas[k * args.op->out_units + i] += oe * input[p * args.op->in_units + k] / args.op->batchsize;
            }
        }
    }
    return;
}

static void fc_op_backward_range_profile(const fc_args *args,
                                         int64_t *d_input_bias_cycles,
                                         int64_t *d_weights_cycles)
{
    int64_t t0 = 0;

    if (args->st_tunits == 0)
    {
        t0 = fc_cycle_count_local();
        for (int p = 0; p < args->op->batchsize; p++)
        {
            for (register int j = 0; j < args->op->out_units; j++)
            {
                register float d_o = args->op->d_output[p * args->op->out_units + j];
                for (register int i = 0; i < args->op->in_units; i++)
                {
                    args->op->d_input[p * args->op->in_units + i] +=
                        args->op->weights[i * args->op->out_units + j] * d_o;
                }
            }
        }

        for (register int j = 0; j < args->op->out_units; j++)
        {
            register float sum = 0.0f;
            for (int p = 0; p < args->op->batchsize; p++)
                sum += args->op->d_output[p * args->op->out_units + j];
            args->op->d_bias[j] = sum / args->op->batchsize;
        }
        int64_t elapsed = fc_cycle_count_local() - t0;
        if (d_input_bias_cycles)
            *d_input_bias_cycles += elapsed;
    }

    t0 = fc_cycle_count_local();
    register float *w_deltas  = args->op->d_weights;
    for (int i = args->st_tunits; i < args->ed_tunits; i++)
    {
        register float *input = args->op->input;
        for (int p = 0; p < args->op->batchsize; p++)
        {
            register float oe = args->op->d_output[p * args->op->out_units + i];
            if (fabsf(oe) < 1e-9f)
                continue;

            for (int k = 0; k < args->op->in_units; k++)
            {
                w_deltas[k * args->op->out_units + i] +=
                    oe * input[p * args->op->in_units + k] / args->op->batchsize;
            }
        }
    }
    int64_t elapsed = fc_cycle_count_local() - t0;
    if (d_weights_cycles)
        *d_weights_cycles += elapsed;
}

void fc_op_backward(fc_op *op)
{
    fc_op_backward_full(op);
}

void fc_op_backward_full(fc_op *op)
{
    if (op->d_weights == NULL || op->d_bias == NULL) {
        fc_op_backward_input_only(op);
        return;
    }

    short tnum = 8; // number of partitions
    if (op->out_units < tnum) {
        fc_args args;
        args.op = op;
        args.st_tunits = 0;
        args.ed_tunits = op->out_units;
        fc_op_backward_range((void *)(&args));
    }else {
        fc_args args[tnum+1];
        short internal = ceil(1.0 * op->out_units / tnum);

        for (int p = 0; p < tnum; p++)
        {
            args[p].op = op;
            args[p].st_tunits = p*internal;
            args[p].ed_tunits = MIN(args[p].st_tunits+internal, op->out_units);
            if (args[p].st_tunits >= op->out_units) break;
            fc_op_backward_range((void *)(&args[p]));
        }
    }

}

void fc_op_backward_full_profile(fc_op *op, fc_backward_cycle_breakdown *cycles)
{
    if (cycles) {
        cycles->d_input_bias_cycles = 0;
        cycles->d_weights_cycles = 0;
    }

    if (op->d_weights == NULL || op->d_bias == NULL) {
        fc_op_backward_input_only(op);
        return;
    }

    short tnum = 8;
    if (op->out_units < tnum) {
        fc_args args;
        args.op = op;
        args.st_tunits = 0;
        args.ed_tunits = op->out_units;
        fc_op_backward_range_profile(&args,
                                     cycles ? &cycles->d_input_bias_cycles : NULL,
                                     cycles ? &cycles->d_weights_cycles : NULL);
    } else {
        fc_args args[tnum + 1];
        short internal = ceil(1.0 * op->out_units / tnum);

        for (int p = 0; p < tnum; p++)
        {
            args[p].op = op;
            args[p].st_tunits = p * internal;
            args[p].ed_tunits = MIN(args[p].st_tunits + internal, op->out_units);
            if (args[p].st_tunits >= op->out_units) break;
            fc_op_backward_range_profile(&args[p],
                                         cycles ? &cycles->d_input_bias_cycles : NULL,
                                         cycles ? &cycles->d_weights_cycles : NULL);
        }
    }
}

void fc_op_backward_input_only(fc_op *op)
{
    // Only propagate gradients to previous layer when this layer is frozen.
    for (int p = 0; p < op->batchsize; p++)
    {
        for (int j = 0; j < op->out_units; j++)
        {
            register float d_o = op->d_output[p * op->out_units + j];
            for (int i = 0; i < op->in_units; i++)
                op->d_input[p * op->in_units + i] += op->weights[i * op->out_units + j] * d_o;
        }
    }

}


void calloc_fc_weights(fc_op *op)
{
    if (op->weights)
        memset(op->weights, 0, (size_t)op->in_units * op->out_units * sizeof(float));
    if (op->bias)
        memset(op->bias, 0, (size_t)op->out_units * sizeof(float));
}

void free_fc_weights(fc_op *op)
{
    (void)op;
}

void calloc_fc_dweights(fc_op *op)
{
    if (op->d_weights)
        memset(op->d_weights, 0, (size_t)op->in_units * op->out_units * sizeof(float));
    if (op->d_bias)
        memset(op->d_bias, 0, (size_t)op->out_units * sizeof(float));
}

void free_fc_dweights(fc_op *op)
{
    (void)op;
}

void save_fc_weights(fc_op *op)
{
    (void)op;
}

void load_fc_weights(fc_op *op, float *w_array, float *b_array)
{
    memcpy(op->weights, w_array, sizeof(float) * op->in_units * op->out_units);
    memcpy(op->bias, b_array, sizeof(float) * op->out_units);
}
