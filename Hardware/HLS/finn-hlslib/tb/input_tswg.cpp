
#include <hls_stream.h>
using namespace hls;
#include "ap_int.h"
#include "bnn-library.h"
#include "input_tswg.h"
// #include "transpose_slidingwindow.h"
// #include "streamtools.h"

void Testbench(stream<ap_uint<IFM_Channels*INPUT_PRECISION> > & in, stream<ap_uint<IFM_Channels*INPUT_PRECISION> > & out, unsigned int numReps)
{
#pragma HLS DATAFLOW
    stream<ap_uint<SIMD*INPUT_PRECISION> > in_simd("in_simd");
    stream<ap_uint<SIMD*INPUT_PRECISION> > out_simd("out_simd");
    StreamingDataWidthConverter_Batch<IFM_Channels*INPUT_PRECISION, SIMD*INPUT_PRECISION, IFMDim*IFMDim>(in, in_simd, numReps);


    TransposeConvolutionInputGenerator<KERNEL_DIM,
    	IFM_Channels,
    	INPUT_PRECISION,
    	IFMDim,
    	OFMDim,
    	SIMD,
    	STRIDE,
		PADDING>(in_simd, out_simd, numReps, ap_resource_dflt());

    StreamingDataWidthConverter_Batch<SIMD*INPUT_PRECISION, IFM_Channels*INPUT_PRECISION, KERNEL_DIM*KERNEL_DIM*OFMDim*OFMDim*IFM_Channels/SIMD>(out_simd, out, numReps);

}


