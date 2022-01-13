#ifndef TRANSPOSESLIDINGWINDOW_H
#define TRANSPOSESLIDINGWINDOW_H

#include "slidingwindow.h"

// =============================== ADD explanation ========================

template<unsigned int ConvKernelDim, 
     unsigned int IFMChannels,
     unsigned int Input_precision,    
     unsigned int IFMDim, 
     unsigned int OFMDim,
     unsigned int SIMD,
     unsigned int Stride,
     unsigned int Padding, 
     typename R>  
void TransposeConvolutionInputGenerator(
    stream<ap_uint<SIMD*Input_precision> > & in,
    stream<ap_uint<SIMD*Input_precision> > & out,
    const unsigned int numReps,
    R const &r) 
{
    CASSERT_DATAFLOW(IFMChannels % SIMD == 0);
    CASSERT_DATAFLOW(Stride >= 1);
    CASSERT_DATAFLOW(Padding <= ConvKernelDim-1);

    // variables for TConv
    const unsigned int TStride = Stride-1;
    const unsigned int TPadding = ConvKernelDim - 1 - Padding;
    const unsigned int TIFMDim = IFMDim + (IFMDim-1)*TStride;
    const unsigned int TIFMPadDim = TIFMDim + 2*TPadding;
    const unsigned int TOFMDim = TIFMPadDim -ConvKernelDim + 1;
    if (TOFMDim != OFMDim)
    {
    	std::cout<< "[ERROR] Output Dimension given and the calculated one does not match. Given = " << OFMDim << ", Calculated = " << TOFMDim << std::endl;
    	exit(-1);
    }
    const unsigned int TIFMLoopBound = TIFMDim + TPadding;
//    std::cout << "Padding = " << TPadding << " Stride = " << TStride << " IFMPadDim = " << TIFMPadDim << " OFMDim = " << TOFMDim << std::endl;
//    std::cout << "IFMLoopBound = " << TIFMLoopBound << " IFMDim = " << TIFMDim << std::endl;

    const unsigned int multiplying_factor = IFMChannels/SIMD;
    const unsigned int number_blocks = ConvKernelDim + 1 ;
    ap_uint<SIMD*Input_precision> inputBuf[number_blocks][TIFMPadDim * multiplying_factor];
#pragma HLS ARRAY_PARTITION variable=inputBuf complete dim=1
    memory_resource(inputBuf, r);
    const unsigned int cycles_write_block = (TOFMDim * ConvKernelDim * ConvKernelDim * multiplying_factor);
    const unsigned int cycles_read_block = TIFMPadDim * multiplying_factor;
    const unsigned int max_cycles = MAX(cycles_write_block,cycles_read_block);
    const unsigned int baseIter = TIFMPadDim * ConvKernelDim * multiplying_factor// Initial buffer
                                + TOFMDim * max_cycles;
    unsigned int counter_internal_block = 0;
    unsigned int current_block_write = 0;
    unsigned int next_block_write = 0;  
    unsigned int current_line = 0;
    unsigned int read_block = 0; 
    unsigned int inp = 0, ofm_y = 0, ofm_x = 0, k_y = 0, k_x = 0, count_simd =0;
#pragma HLS reset variable=inp

    // some variables for padding and expansion
    int inp_i = -1*TPadding;
    int inp_j = -1*TPadding*multiplying_factor;
    int expand_x = 0, expand_y = 0, channel_iter = 0;

    for (unsigned int count_image = 0; count_image < numReps; count_image++) 
    {
        for (unsigned int i = 0; i < baseIter; i++) 
        {
#pragma HLS PIPELINE II=1
            if (inp < TIFMPadDim*ConvKernelDim*multiplying_factor)
            {// Initial buffer of ConvKernelDim lines
                ap_uint<SIMD*Input_precision> inElem;
            	if ( (inp_i < 0) || (inp_j < 0) || (inp_i >= TIFMDim) || (inp_j >= TIFMDim*multiplying_factor))
                	{inElem = 0;}
            	else if ( (expand_x > 0) || (expand_y > 0) )
            	{
            		inElem = 0;
            		expand_x--;
            		expand_y--;

            	}
            	else
            	{
            		inElem = in.read();
            		expand_y--;
            		if (channel_iter == multiplying_factor-1)
            			{expand_x = TStride*multiplying_factor;}
            	}
                inputBuf[current_block_write][current_line] = inElem;

                channel_iter++;
                if (channel_iter == multiplying_factor)
                	{channel_iter = 0;}
        		inp_j++;
                if (inp_j == TIFMLoopBound*multiplying_factor)
                {
                    inp_j = -1*TPadding*multiplying_factor;
                    inp_i++;
                    if (inp_i == TIFMLoopBound)
                        {inp_i = -1*TPadding;}
                    if (expand_y < 0)
                    	{expand_y = TIFMDim*TStride*multiplying_factor;}
                }

                current_line++;
                inp++;
                if (current_line == TIFMPadDim * multiplying_factor )
                {
                    current_line = 0;
                    current_block_write++;
                    if (current_block_write == number_blocks) 
                        {current_block_write=0;}
                    read_block++;
                    counter_internal_block = 0;
                }
            } 
            else 
            {
                if (counter_internal_block < cycles_write_block-1) 
                { // We are writing output, MMV IFMChan per cycle
                    unsigned int current_block_read = (current_block_write + 1 + k_y);
                    if (current_block_read >= number_blocks) 
                        {current_block_read-= number_blocks;}
                    unsigned int current_line_in_block = (ofm_x+k_x)*multiplying_factor + count_simd;
                    ap_uint<SIMD*Input_precision> outElem = inputBuf[current_block_read][(current_line_in_block)];
                    out.write(outElem);
                    count_simd++;
                    if (count_simd == multiplying_factor) 
                    {
                        count_simd=0;         
                        k_x++;
                        if (k_x == ConvKernelDim) 
                        {
                            k_x = 0;
                            k_y++;
                            if (k_y == ConvKernelDim) 
                            {
                                k_y = 0;
                                ofm_x ++;
                                if (ofm_x == TOFMDim)
                                {
                                    ofm_x = 0;
                                    ofm_y++;
                                    if (ofm_y == TOFMDim)
                                    {
                                        ofm_y = 0;
                                        inp = 0;
                                    }
                                }
                            }
                        }
                    }
                }
                if ((counter_internal_block < cycles_read_block-1) && (read_block<TIFMPadDim))
                { // In parallel we write in the buffer, in the current block write if we still need to
                    ap_uint<SIMD*Input_precision> inElem;
                	if ( (inp_i < 0) || (inp_j < 0) || (inp_i >= TIFMDim) || (inp_j >= TIFMDim*multiplying_factor))
                    	{inElem = 0;}
                	else if ( (expand_x > 0) || (expand_y > 0) )
                	{
                		inElem = 0;
                		expand_x--;
                		expand_y--;
                	}
                	else
                	{
                		inElem = in.read();
                		expand_y--;
                		if (channel_iter == multiplying_factor-1)
                			{expand_x = TStride*multiplying_factor;}
                	}
                    inputBuf[current_block_write][current_line] = inElem;
#pragma AP dependence variable=inputBuf intra false
#pragma AP dependence variable=inputBuf inter false
                    channel_iter++;
                    if (channel_iter == multiplying_factor)
                    	{channel_iter = 0;}
#pragma AP dependence variable=channel_iter intra false
            		inp_j++;
                    if (inp_j == TIFMLoopBound*multiplying_factor)
                    {
                        inp_j = -1*TPadding*multiplying_factor;
                        inp_i++;
                        if (inp_i == TIFMLoopBound)
                            {inp_i = -1*TPadding;}
                        if (expand_y < 0)
                        	{expand_y = TIFMDim*TStride*multiplying_factor;}
                    }
                    current_line++;
                    if (current_line == TIFMPadDim * multiplying_factor)
                    {// We read the whole block, we change the next block in which we want to we
                        // We filled up a block, let's not read until
                        current_line = 0;
                        read_block++;
                        current_block_write++;
                        if (current_block_write == number_blocks) 
                            {current_block_write=0;}
#pragma AP dependence variable=current_block_write intra false  
                    }
                }
                counter_internal_block++; // = (counter_internal_block +1) % max_cycles;
                if (counter_internal_block == (max_cycles-1)) 
                    {counter_internal_block = 0;}
            }
        } // End base_iter
        read_block = 0, expand_x = 0, expand_y = 0;
    } // End count_image
} // End generator

#endif
