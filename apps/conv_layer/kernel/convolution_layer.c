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
#include "fconv3d.h"
#ifndef MIN
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#endif

static float conv1_xcol_scratch[32 * 32 * (3 * 3 * 3)];
static float conv2_xcol_scratch[16 * 16 * (64 * 3 * 3)];
static float conv3_xcol_scratch[8 * 8 * (128 * 3 * 3)];
static float conv4_xcol_scratch[8 * 8 * (256 * 3 * 3)];
static float conv5_input_col_full[CONV5_INPUT_COL_SIZE];

static float conv_t_dweights_scratch[CONV_MAX_T_DWEIGHTS];
static float conv_d_out_copy_scratch[CONV_MAX_DOCOPY];
static float conv_d_x_col_scratch[CONV_MAX_DXCOL];
static float conv_weights_t_scratch[CONV_MAX_T_DWEIGHTS];
static float conv3x3_filter_scratch[CONV_MAX_IKK];

static float *conv_forward_xcol_ptr(conv_op *op, short batch_id);
static void img2col(const float *img, float *col, const conv_op *op);
static void conv_op_forward_3x3_ara(conv_op *op);

static void pack_conv3x3_filter(const conv_op *op, int oc, float *dst)
{
    int idx = 0;
    for (int ic = 0; ic < op->in_channels; ic++) {
        for (int ky = 0; ky < op->kernel_size; ky++) {
            for (int kx = 0; kx < op->kernel_size; kx++) {
                int w_idx = (ic * op->kernel_size * op->kernel_size + ky * op->kernel_size + kx) * op->out_channels + oc;
                dst[idx++] = op->weights[w_idx];
            }
        }
    }
}

static int conv_can_use_3x3_ara(const conv_op *op)
{
    if (op->kernel_size != 3 || op->stride != 1)
        return 0;
    if (op->in_w != op->out_w + 2 || op->in_h != op->out_h + 2)
        return 0;
    return 1;
}

static void conv_op_forward_3x3_ara(conv_op *op)
{
    int out_plane = op->out_w * op->out_h;
    int in_channels = op->in_channels;
    int out_channels = op->out_channels;

    for (int b = 0; b < op->batchsize; b++) {
        const float *input_b = op->input + b * op->in_units;
        float *output_b = op->output + b * op->out_units;

        for (int oc = 0; oc < out_channels; oc++) {
            pack_conv3x3_filter(op, oc, conv3x3_filter_scratch);
            float *out_oc = output_b + oc * out_plane;
            fconv3d_CHx3x3_f32(out_oc, input_b, conv3x3_filter_scratch,
                               op->out_h, op->out_w, in_channels, op->bias[oc]);
        }

        if (op->input_col != NULL) {
            float *x_col = conv_forward_xcol_ptr(op, (short)b);
            img2col(input_b, x_col, op);
        }
    }
}

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


static void img2col(const float *img, float *col, const conv_op *op) // not classic. row-major patches
{
    int iwih = op->in_w * op->in_h; 
    int kk   = op->kernel_size * op->kernel_size; //number of pixels in a channel of a kernel
    int ikk  = op->in_channels * kk; //total number of pixels in a kernel

    //st_x,y == sum of stride x,y


    for (int in_c = 0; in_c < op->in_channels; in_c++)
    {
        int out_y = 0;
        for (int st_y = 0; st_y < op->out_h * op->stride; st_y += op->stride, out_y++)
        {
            int out_x = 0;
            for (int st_x = 0; st_x < op->out_w * op->stride; st_x += op->stride, out_x++) //we move per stride
            {

                int patch_idx = out_y * op->out_w + out_x; // row-major patch index
                int x_col_offset = patch_idx * ikk + in_c * kk; //+0
                // in_c*kk==prosdiorismos pixel se sygkekrimeno channel
                //patch_idx*ikk=poy ksekinaei h nea grammh pou antistoixei
                //so in those 2 lines we set which is the patch of the image, we want to look at

                for (int j = 0; j < op->kernel_size; j++)
                {
                    for (int i = 0; i < op->kernel_size; i++)
                    {
                        int input_offset = (st_x + i) + (st_y + j) * op->in_w + in_c * iwih;
                        col[x_col_offset] = img[input_offset];
                        x_col_offset++;
                    }
                }
            }
        }
    }
} //so we destroy data locality in this algorithm in order to have data locality in GEMM


static void conv_op_forward_single(conv_op *op, short batch_id)
{
    float *x_col = conv_forward_xcol_ptr(op, batch_id);
    float *t_input  = op->input + batch_id * op->in_units;
    float *t_output = op->output + batch_id * op->out_units;
    int ikk  = op->in_channels * op->kernel_size * op->kernel_size;
    int owoh = op->out_w * op->out_h;
    // 
    // >>>>>>>shape<<<<<<<
    //  
    // t_input    [ic,ih,iw]
    // x_col      [owoh,ikk]
    // weights    [ikk,oc]
    // t_output   [oc,oh,ow]
    // >>>>>>>>>>>>>>>>>>>
    //
    img2col(t_input, x_col, op);
    matrix_multiply(x_col, op->weights, t_output, owoh, ikk, op->out_channels);
    matrix_transpose(t_output, owoh, op->out_channels);

    register int o_offset=0;
    for (int i = 0; i < op->out_channels; i++)
    {
        register float tmp = op->bias[i];
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
    
    conv_op_forward_3x3_ara(op);
    return;


}

void conv_op_forward_im2col(conv_op *op)
{
    if (op->layer_id == 5 && op->input_col == NULL) {
        printf_("Error: conv_op_forward_im2col requires input_col for layer_id=5\n");
        exit(1);
    }

    for (int p = 0; p < op->batchsize; p++)
    {
        conv_op_forward_single(op, (short)p);
    }
}


static void col2img(const float *col, float *img, const conv_op *op)
{
    int iwih = op->in_w * op->in_h;
    int kk   = op->kernel_size * op->kernel_size;
    int ikk  = op->in_channels * kk;

    int out_y = 0;
    for (int st_y = 0; st_y < op->out_h * op->stride; st_y += op->stride, out_y++)
    {
        int out_x = 0;
        for (int st_x = 0; st_x < op->out_w * op->stride; st_x += op->stride, out_x++)
        {
            for (int in_c = 0; in_c < op->in_channels; in_c++)
            {
                // Ο ίδιος ασφαλής υπολογισμός
                int patch_idx = out_y * op->out_w + out_x;
                int x_col_offset = patch_idx * ikk + in_c * kk;

                for (int j = 0; j < op->kernel_size; j++)
                {
                    for (int i = 0; i < op->kernel_size; i++)
                    {
                        int input_offset = (st_x + i) + (st_y + j) * op->in_w + in_c * iwih;
                        img[input_offset] += col[x_col_offset];
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
    int oc = op->out_channels;
    int ikk = op->in_channels * op->kernel_size * op->kernel_size;
    int owoh = op->out_w * op->out_h;

    float *t_d_weights = conv_t_dweights_scratch;
    
    // ΕΠΑΝΑΧΡΗΣΙΜΟΠΟΙΗΣΗ ΜΝΗΜΗΣ:
    // Χρησιμοποιούμε προσωρινά το conv_weights_t_scratch ως accumulator.
    // Είναι ελεύθερο σε αυτή τη φάση και έχει ακριβώς το σωστό μέγεθος (ikk * oc).
    float *batch_d_weights_acc = conv_weights_t_scratch;
    memset(batch_d_weights_acc, 0, ikk * oc * sizeof(float));

    for (int p = 0; p < op->batchsize; p++)
    {
        // Επαναφέρω το memset σε περίπτωση που η matrix_multiply κάνει +=
        memset(t_d_weights, 0, oc * ikk * sizeof(float));
        matrix_multiply(op->d_output + p * oc * owoh, op->input_col + p * owoh * ikk, t_d_weights, oc, owoh, ikk);

        for (int j = 0; j < oc; j++)
        {
            for (int i = 0; i < ikk; i++)
                batch_d_weights_acc[i * oc + j] += t_d_weights[j * ikk + i]; // Αθροίζουμε ΠΡΙΝ τη διαίρεση
        }
    }

    // Προσθέτουμε τα συνολικά gradients στο πραγματικό d_weights, κάνοντας ΕΔΩ τη διαίρεση.
    for (int i = 0; i < ikk * oc; i++) 
    {
        op->d_weights[i] += batch_d_weights_acc[i] / op->batchsize;
    }

    // --- Τέλος χρήσης του batch_d_weights_acc ---

    // Bias υπολογισμός
    for (int i = 0; i < op->out_channels; i++)
    {
        float tmp = 0.0f;
        for (int p = 0; p < op->batchsize; p++)
            for (int s = i * owoh; s < (i + 1) * owoh; s++)
                tmp += op->d_output[p * oc * owoh + s];
        op->d_bias[i] = tmp / op->batchsize;
    }

    // Delta input υπολογισμός
    // Τώρα το conv_weights_t_scratch χρησιμοποιείται για τον κανονικό του σκοπό.
    // Το memcpy που ακολουθεί γράφει πάνω στα προηγούμενα δεδομένα (accumulator) με ασφάλεια.
    float *weights_T = conv_weights_t_scratch;
    memcpy(weights_T, op->weights, ikk * oc * sizeof(float));
    matrix_transpose(weights_T, ikk, oc);

    float *d_out_copy = conv_d_out_copy_scratch;
    float *d_x_col = conv_d_x_col_scratch;

    for (int p = 0; p < op->batchsize; p++)
    {
        memcpy(d_out_copy, op->d_output + p * oc * owoh, oc * owoh * sizeof(float));
        matrix_transpose(d_out_copy, oc, owoh);
        
        // Επαναφέρω το memset
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
