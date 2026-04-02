//
// File:        convolution_layer.c
// Description: Implementation of convolution layer
// Author:      Haris Wang
//
#include <stdlib.h>
#include <math.h>
#include <string.h>
#ifdef SPIKE
#include "printf.h"
#elif defined ARA_LINUX
#include <stdio.h>
#else
#include "printf.h"
#endif
#include "convolution_layer.h"
#include "matrix.h"
#define MIN(a,b) (((a) < (b)) ? (a) : (b))

static float conv1_xcol_scratch[32 * 32 * (3 * 3 * 3)];
static float conv2_xcol_scratch[16 * 16 * (64 * 3 * 3)];
static float conv3_xcol_scratch[8 * 8 * (128 * 3 * 3)];
static float conv4_xcol_scratch[8 * 8 * (256 * 3 * 3)];
static float conv5_input_col_full[CONV5_INPUT_COL_SIZE];

static float conv_t_dweights_scratch[CONV_MAX_T_DWEIGHTS];
static float conv_d_out_copy_scratch[CONV_MAX_DOCOPY];
static float conv_d_x_col_scratch[CONV_MAX_DXCOL];
static float conv_weights_t_scratch[CONV_MAX_T_DWEIGHTS];

static float *conv_forward_xcol_ptr(conv_op *op, short batch_id)
{
    int col_size_per_image = (op->in_channels * op->kernel_size * op->kernel_size) * (op->out_w * op->out_h);
    if (op->layer_id == 5)
        return op->input_col + batch_id * col_size_per_image;

    switch (op->layer_id) {
        case 1: return conv1_xcol_scratch;
        case 2: return conv2_xcol_scratch;
        case 3: return conv3_xcol_scratch;
        case 4: return conv4_xcol_scratch;
        default:
            printf_("Error: invalid conv layer_id=%d\n", op->layer_id);
            exit(1);
    }
}


typedef struct conv_args{
    conv_op *op;
    short batch_id;
    short st_tunits;
    short ed_tunits;
} conv_args;

static void img2col(const float *img, float *col, const conv_op *op) //not classic. transposed
{
    int iwih = op->in_w * op->in_h; 
    int kk   = op->kernel_size * op->kernel_size; //number of pixels in a channel of a kernel
    int ikk  = op->in_channels * kk; //total number of pixels in a kernel

    //st_x,y == sum of stride x,y


    for (int in_c = 0; in_c < op->in_channels; in_c++)
    {
        int out_x = 0;
        for (int st_x = 0; st_x < op->out_w * op->stride; st_x += op->stride, out_x++) //we move per stride
        {
            int out_y = 0;
            for (int st_y = 0; st_y < op->out_h * op->stride; st_y += op->stride, out_y++)
            {

                int patch_idx = out_x * op->out_h + out_y; //indexing of patches, vertically
                int x_col_offset = patch_idx * ikk + in_c * kk; //+0
                // in_c*kk==prosdiorismos pixel se sygkekrimeno channel
                //patch_idx*ikk=poy ksekinaei h nea grammh pou antistoixei
                //so in those 2 lines we set which is the patch of the image, we want to look at

                for (int j = 0; j < op->kernel_size; j++)
                {
                    for (int i = 0; i < op->kernel_size; i++)
                    {
                        if (st_x + i < op->in_w && st_y + j < op->in_h)
                        {
                            int input_offset = (st_x + i) + (st_y + j) * op->in_w + in_c * iwih; //h thesh ston 3d pinaka ths eikonas
                            //index=x⋅(Y⋅Z)+y⋅Z+z
                            col[x_col_offset] = img[input_offset];
                        }
                        else
                        {
                            col[x_col_offset] = 0.0f;
                        }
                        x_col_offset++;
                    }
                }
            }
        }
    }
} //so we destroy data locality in this algorithm in order to have data locality in GEMM


static void conv_op_forward_single(void *argv)
{
    
    /**
     * pthread conv_op_forward
     * 
     * argv is a pointer in struct type conv_args
     * */
    conv_args cp;
    memcpy(&cp, (conv_args *)argv, sizeof(conv_args));
    float *x_col = conv_forward_xcol_ptr(cp.op, cp.batch_id);
    // float *x_col    = cp.op->input_col + cp.batch_id * cp.op->in_units; i changed it...
    float *t_input  = cp.op->input + cp.batch_id * cp.op->in_units; //take the image that is for the image that your thread uses
    float *t_output = cp.op->output + cp.batch_id * cp.op->out_units; // store the output of the convolution in the right, channel
    int ikk  = cp.op->in_channels * cp.op->kernel_size * cp.op->kernel_size;  //weight dimension j
    int owoh = cp.op->out_w * cp.op->out_h;
    // 
    // >>>>>>>shape<<<<<<<
    //  
    // t_input    [ic,ih,iw]
    // x_col      [owoh,ikk]
    // weights    [ikk,oc]
    // t_output   [oc,oh,ow]
    // >>>>>>>>>>>>>>>>>>>
    //
    img2col(t_input, x_col, cp.op);
    matrix_multiply(x_col, cp.op->weights, t_output, owoh, ikk, cp.op->out_channels); //output[owoh,oc]
    matrix_transpose(t_output, owoh, cp.op->out_channels); //output[oc,owoh]

    register int o_offset=0;
    for (int i = 0; i < cp.op->out_channels; i++)
    {
        register float tmp = cp.op->bias[i];
        while (o_offset < (i+1)*owoh)
        {
            t_output[o_offset++] += tmp;
        }
    }


    return;
}

/*

typedef struct conv_args{
    conv_op *op;
    short batch_id;
    short st_tunits;
    short ed_tunits;
} conv_args;
*/


void conv_op_forward(conv_op *op)
{
    /**
     * conv2d forward
     * 
     * Input:
     *      op->input
     *      op->weights
     *      op->bias
     * Output:
     *      op->output
     * */
    if (op->layer_id == 5) {
        if (op->batchsize > ALEXNET_STATIC_MAX_BATCH) {
            printf_("Error: conv5 batchsize %d exceeds static max %d\n", op->batchsize, ALEXNET_STATIC_MAX_BATCH);
            exit(1);
        }
        op->input_col = conv5_input_col_full;
        memset(op->input_col, 0,
               (size_t)op->batchsize * (size_t)(op->in_channels * op->kernel_size * op->kernel_size) *
               (size_t)(op->out_w * op->out_h) * sizeof(float));
    } else {
        op->input_col = NULL;
    }
    conv_args args[op->batchsize+1]; //each image in the batch has its own struct
    for (int p = 0; p < op->batchsize; p++)
    {
        args[p].op = op;
        args[p].batch_id = p;
        conv_op_forward_single((void *)(&args[p])); // sequential over batch images
    } //tosa threads osa kai ta images sto batch

}


static void col2img(const float *col, float *img, const conv_op *op)
{
    int iwih = op->in_w * op->in_h;
    int kk   = op->kernel_size * op->kernel_size;
    int ikk  = op->in_channels * kk;

    int out_x = 0;
    for (int st_x = 0; st_x < op->out_w * op->stride; st_x += op->stride, out_x++)
    {
        int out_y = 0;
        for (int st_y = 0; st_y < op->out_h * op->stride; st_y += op->stride, out_y++)
        {
            for (int in_c = 0; in_c < op->in_channels; in_c++)
            {
                // Ο ίδιος ασφαλής υπολογισμός
                int patch_idx = out_x * op->out_h + out_y;
                int x_col_offset = patch_idx * ikk + in_c * kk;

                for (int j = 0; j < op->kernel_size; j++)
                {
                    for (int i = 0; i < op->kernel_size; i++)
                    {
                        if (st_x + i < op->in_w && st_y + j < op->in_h)
                        {
                            int input_offset = (st_x + i) + (st_y + j) * op->in_w + in_c * iwih;
                            img[input_offset] += col[x_col_offset]; 
                        }
                        x_col_offset++;
                    }
                }
            }
        }
    }
}


void conv_op_backward(conv_op *op)
{
    conv_op_backward_full(op);
}

void conv_op_backward_full(conv_op *op)
{
    /**
     * conv2d backward
     * 
     * Input:
     *      op->d_output
     * Output:
     *      op->d_weights
     *      op->d_bias
     *      op->d_input
     * */
    int oc = op->out_channels;
    int ikk = op->in_channels * op->kernel_size * op->kernel_size;
    int owoh = op->out_w * op->out_h;

    // calculate delta_weights using per-sample d_output

    float *t_d_weights = conv_t_dweights_scratch;
    for (int p = 0; p < op->batchsize; p++)
    {
        memset(t_d_weights, 0, oc * ikk * sizeof(float));
        matrix_multiply(op->d_output + p * oc * owoh, op->input_col + p * owoh * ikk, t_d_weights, oc, owoh, ikk);

        // t_d_weights[oc, internal] = dW^T. Must store as dW[ikk, oc] to match weights layout.
        for (int j = 0; j < oc; j++)
        {
            for (int i = 0; i < ikk; i++)
                op->d_weights[i * oc + j] += t_d_weights[j * ikk + i] / op->batchsize;
        }
    }

    // calculate delta_bias averaged across batch
    for (int i = 0; i < op->out_channels; i++)
    {
        register float tmp = 0.0f;
        for (int p = 0; p < op->batchsize; p++)
            for (int s = i * owoh; s < (i + 1) * owoh; s++)
                tmp += op->d_output[p * oc * owoh + s];
        op->d_bias[i] = tmp / op->batchsize;
    }

    // calculate delta_input per sample
    if (ikk * oc > CONV_MAX_T_DWEIGHTS) {
        printf_("Error: conv weights transpose workspace overflow (%d)\n", ikk * oc);
        exit(1);
    }
    float *weights_T = conv_weights_t_scratch;
    memcpy(weights_T, op->weights, ikk * oc * sizeof(float));
    matrix_transpose(weights_T, ikk, oc);

    float *d_out_copy = conv_d_out_copy_scratch;
    float *d_x_col = conv_d_x_col_scratch;

    for (int p = 0; p < op->batchsize; p++)
    {
        memcpy(d_out_copy, op->d_output + p * oc * owoh, oc * owoh * sizeof(float));
        matrix_transpose(d_out_copy, oc, owoh);
        memset(d_x_col, 0, ikk * owoh * sizeof(float));
        matrix_multiply(d_out_copy, weights_T, d_x_col, owoh, oc, ikk);
        col2img(d_x_col, op->d_input + p * op->in_units, op);
    }

    op->input_col = NULL;

}

void conv_op_backward_input_only(conv_op *op)
{
    // Only propagate d_input for frozen convolution layers.
    int oc   = op->out_channels;
    int ikk  = op->in_channels * op->kernel_size * op->kernel_size;
    int owoh = op->out_w * op->out_h;
    if (ikk * oc > CONV_MAX_T_DWEIGHTS) {
        printf_("Error: conv weights transpose workspace overflow (%d)\n", ikk * oc);
        exit(1);
    }
    float *weights_T = conv_weights_t_scratch;
    memcpy(weights_T, op->weights, ikk * oc * sizeof(float));
    matrix_transpose(weights_T, ikk, oc);

    float *d_out_copy = conv_d_out_copy_scratch;
    float *d_x_col = conv_d_x_col_scratch;

    for (int p = 0; p < op->batchsize; p++)
    {
        memcpy(d_out_copy, op->d_output + p * oc * owoh, oc * owoh * sizeof(float));
        matrix_transpose(d_out_copy, oc, owoh);
        memset(d_x_col, 0, ikk * owoh * sizeof(float));
        matrix_multiply(d_out_copy, weights_T, d_x_col, owoh, oc, ikk);
        col2img(d_x_col, op->d_input + p * op->in_units, op);
    }

    op->input_col = NULL;
}

void calloc_conv_weights(conv_op *op)
{
    if (op->weights)
        memset(op->weights, 0, (size_t)op->out_channels * op->in_channels * op->kernel_size * op->kernel_size * sizeof(float));
    if (op->bias)
        memset(op->bias, 0, (size_t)op->out_channels * sizeof(float));
}

void free_conv_weights(conv_op *op)
{
    (void)op;
}

void calloc_conv_dweights(conv_op *op)
{
    if (op->d_weights)
        memset(op->d_weights, 0, (size_t)op->out_channels * op->in_channels * op->kernel_size * op->kernel_size * sizeof(float));
    if (op->d_bias)
        memset(op->d_bias, 0, (size_t)op->out_channels * sizeof(float));
}

void free_conv_dweights(conv_op *op)
{
    (void)op;
}

void save_conv_weights(conv_op *op)
{
    (void)op;
}


void load_conv_weights(conv_op *op, float *w_array, float *b_array)
{
    memcpy(op->weights, w_array,
           sizeof(float) * op->out_channels * op->in_channels * op->kernel_size * op->kernel_size);
    memcpy(op->bias, b_array, sizeof(float) * op->out_channels);
}
