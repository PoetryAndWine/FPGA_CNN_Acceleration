/**********
Copyright (c) 2017, Xilinx, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********/

/*******************************************************************************

SDx Key Concept :
    
    This is a CNN (Convolutional Neural Network) convolution layer based example
    to showcase the effectiveness of using multiple compute units when the base
    algorithm consists of multiple nested loops with large loop count. The main
    aim of this example is to help developer to overcome clock frequency issues
    and achieve better performance.

*******************************************************************************/

/*

Kernel Description (Good Example) :
   
    This example uses multiple compute units & wide range of work items which
    process Output filters when convolution operation is triggered. It also
    presents the uniform local memory alignment when multiple work_group/compute
    units are used. 
    
    Note : Naive version (Bad Example) version is in the file cnn_convolution_bad.cl
    
    Arguments :
    
        int *image   (input )  --> Input Image    
        int *weight  (input )  --> Input Weights  
        int *out     (output)  --> Output filters 
        int  size    (input )  --> Output size

    Kernel Configuration :
        
        1. Output Channels    = 256
        2. Work Groups        = Number of Compute Units
        3. Work Items         = One per Work_Group (Compute Unit) 

        -----------------------------------------------------
        | Parameter     | Value |   Description             |
        -----------------------------------------------------
        | Channels      | 96    | #Input Channels           |                            
        -----------------------------------------------------
        | IHeight       | 27    | Input Image Height        |
        -----------------------------------------------------
        | IWidth        | 27    | Input Image Width         |
        -----------------------------------------------------
        | Window        | 5     | Convolution Window Size   |
        -----------------------------------------------------
        | Stride        | 1     | Convolution Stride        |
        -----------------------------------------------------
        | Padding       | 2     | Convolution Image Padding |
        -----------------------------------------------------
        | OutputFilters | 256   | Output Filters/Images     |
        -----------------------------------------------------
        | OHeight       | 27    | Output Image Height       |
        -----------------------------------------------------
        | OWidth        | 27    | Output Image Width        |
        -----------------------------------------------------


    Memory Usage (Local Buffer (Per Work_Group / Compute Unit)):
        
        1. Image    ~ (IHeight x IWidth x Channels):[2.84 x 96 KB]
        2. Weights  ~ (Channels x Window x Window):[96 x 0.09 KB]
        3. Output   ~ (OHeight x OWidth):[2.84 KB]

    Reference : 
             
        To understand Convolution Layer of a CNN better please refer to
        website below (Convolution Demo Animation in the link).
                                                     
        Link: http://cs231n.github.io/convolutional-networks/
*/

#include "defns.h"

void __attribute__((always_inline)) copy_weight(__global int *weight, int wgt_lcl[WInChan][WSize * WSize], int output)
{
    int stride = output * WInChan * WSize * WSize;

    __attribute__((xcl_pipeline_loop))
    readWt: for(int itr = 0, i = 0, j = 0; itr < WInChan * WSize * WSize; itr++,j++) {
        if(j == WSize * WSize) {j = 0; i++;}
        wgt_lcl[i][j] = weight[stride+itr];
    }
}

void __attribute__((always_inline)) copy_output(__global int *out, int out_lcl[OSize * OSize], int output)
{
    int stride = output * OSize * OSize;
 
    __attribute__((xcl_pipeline_loop))
    writeOut: for(int itr = 0; itr < OSize * OSize; itr++) {
        out[stride + itr] = out_lcl[itr];
    }
}

void __attribute__((always_inline))
    convolution_operation(int img_lcl[IChan][ISize * ISize], int wgt_lcl[WInChan][WSize * WSize], int out_lcl[OSize * OSize],int output, int y, int x, int i_chan)
{
    short acc[IChan][WSize][WSize]
    __attribute__((xcl_array_partition(complete,1)));

    int xVal_base = x * Stride - Padding;
    int yVal = y * Stride - Padding;

    __attribute__((xcl_pipeline_loop))
    convYaxis: for(int i = 0; i < WSize; i++,yVal++){
        convXaxis: for(int j = 0, xVal = xVal_base ; j < WSize; j++, xVal++){
            convInchan: for(int input = 0; input < IChan; input++) {

                
                if(yVal >= 0 && yVal < ISize && xVal >= 0 && xVal < ISize) {
                    acc[input][i][j] =  (short) img_lcl[input][yVal * ISize + xVal] *
                                        (short) wgt_lcl[input][i * WSize + j];
                }
                else {
                    acc[input][i][j] = 0;
                }
            }
        }
    }
    
    
    short sum = 0;
    __attribute__((xcl_pipeline_loop))
    accJ: for(int j = 0; j < WSize;j++)
        accK: for(int k = 0; k < WSize; k++)
            accI: for(int i = 0; i < IChan; i++)
                sum += acc[i][j][k];

    
    out_lcl[y * OSize + x] = sum;
}

__kernel __attribute__ ((reqd_work_group_size(1, 1, 1)))
void good_cnn(
        const __global int *image,      
        const __global int *weights,    
        __global int *out,              
        int size,                       
        int i_chan,                    
        int o_chan                      
    )
{
    int global_id      = get_global_id(0);         
    int global_threads = get_global_size(0);       

    int img_lcl[IChan][ISize * ISize] __attribute__((xcl_array_partition(complete,1)));

    int out_lcl[OSize * OSize];

    int wgt_lcl[WInChan][WSize * WSize] __attribute__((xcl_array_partition(complete,1)));
    
    __attribute__((xcl_pipeline_loop))
    imgcopy:for(int itr = 0, i = 0, j = 0; itr < IChan * ISize * ISize; itr++, j++){
        if(j == ISize * ISize) {j = 0; i++;}
            img_lcl[i][j] = image[itr];
    }

    int thread_work_start = global_id * o_chan / global_threads;
    int thread_work_end   = (global_id + 1) * o_chan / global_threads;
    
   
    outthread:for(int output = thread_work_start; output < thread_work_end; output++) {
            
        
        copy_weight(weights, wgt_lcl, output);

        outYaxis:for(int y = 0; y < OSize; y++) {
            outXaxis:for(int x = 0; x < OSize; x++) {
               
               convolution_operation(img_lcl, wgt_lcl, out_lcl, output, y, x, i_chan);
            }
        }
        
     
        copy_output(out, out_lcl, output);
    }

    return;
}
