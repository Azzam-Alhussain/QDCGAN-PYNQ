 
#include "config.h"
#include "bnn-library.h"
#include "weights.hpp"
#include "activations.hpp"
#include "interpret.hpp"
#include "mvau.hpp"


static FixedPointWeights<L0_SIMD, ap_fixed<L0_WPI+L0_WPF,L0_WPI,AP_TRN,AP_WRAP>, L0_PE, L0_WMEM>   weights0;
static FixedPointWeights<L1_SIMD, ap_fixed<L1_WPI+L1_WPF,L1_WPI,AP_TRN,AP_WRAP>, L1_PE, L1_WMEM>   weights1;
static FixedPointWeights<L2_SIMD, ap_fixed<L2_WPI+L2_WPF,L2_WPI,AP_TRN,AP_WRAP>, L2_PE, L2_WMEM>   weights2;
static FixedPointWeights<L3_SIMD, ap_fixed<L3_WPI+L3_WPF,L3_WPI,AP_TRN,AP_WRAP>, L3_PE, L3_WMEM>   weights3;
static FixedPointWeights<L4_SIMD, ap_fixed<L4_WPI+L4_WPF,L4_WPI,AP_TRN,AP_WRAP>, L3_PE, L3_WMEM>   weights4;

static ReLUActivation<1,1,15,ap_fixed<32, 20, AP_TRN, AP_WRAP>,ap_uint<4>,0> threshs0 = {
{
{
{
0.03125,
0.09375,
0.15625,
0.21875,
0.28125,
0.34375,
0.40625,
0.46875,
0.53125,
0.59375,
0.65625,
0.71875,
0.78125,
0.84375,
0.90625
}
}
}
};

void DoMemInit(unsigned int targetLayer, unsigned int targetMem, unsigned int targetInd, unsigned int targetThresh, ap_uint<64> val) {
  switch (targetLayer) {
    case 0:
      weights0.m_weights[targetMem][targetInd] = val;
      break;
    case 2:
      weights1.m_weights[targetMem][targetInd] = val;
      break;
    case 4:
      weights2.m_weights[targetMem][targetInd] = val;
      break;
    case 6:
      weights3.m_weights[targetMem][targetInd] = val;
      break;
    case 8:
      weights4.m_weights[targetMem][targetInd] = val;
      break;
    case 9:
      break;
  }
}

void DoCompute(ap_uint<64> *in, ap_uint<64> *out, const unsigned int numReps) 
{
#pragma HLS DATAFLOW
  stream<ap_uint<64>> inter0("DoCompute.inter0");
#pragma HLS STREAM variable=inter0 depth=2

  stream<ap_uint<256>> inter1("DoCompute.inter1");
#pragma HLS STREAM variable=inter1 depth=16

  stream<ap_uint<256>> inter2("DoCompute.inter2");
#pragma HLS STREAM variable=inter2 depth=32

  stream<ap_uint<256>> inter3("DoCompute.inter3");
#pragma HLS STREAM variable=inter3 depth=16

  stream<ap_uint<256>> inter4("DoCompute.inter4");
#pragma HLS STREAM variable=inter4 depth=16

  stream<ap_uint<192>> inter5("DoCompute.inter5");
#pragma HLS STREAM variable=inter5 depth=16

  stream<ap_uint<64>> memOutStream("DoCompute.memOutStream");

  const unsigned int inBits = IMG_DIM * IMG_DIM * IMG_CH * 8;
  const unsigned int outBits = L4_OFM_DIM * L4_OFM_DIM * L4_OFM_CH * 8;

  Mem2Stream_Batch<64, inBits / 8>(in, inter0, numReps);

  TransposeConvLayer_Batch
  <
  L0_K, L0_IFM_CH, L0_IFM_DIM, L0_OFM_CH, L0_OFM_DIM, 1, 0, L0_SIMD, L0_PE,
  Slice<ap_fixed<8, 1, AP_TRN, AP_SAT>>, Slice<ap_uint<4>>, Identity
  >
  (inter0, inter1, weights0, threshs0, numReps, ap_resource_lut());


  TransposeConvLayer_Batch
  <
  L1_K, L1_IFM_CH, L1_IFM_DIM, L1_OFM_CH, L1_OFM_DIM, 2, 1, L1_SIMD, L1_PE,
  Slice<ap_ufixed<4, 0, AP_TRN, AP_WRAP>>, Slice<ap_uint<4>>, Identity
  >
  (inter1, inter2, weights1, threshs0, numReps, ap_resource_lut());

 TransposeConvLayer_Batch
 <
 L2_K, L2_IFM_CH, L2_IFM_DIM, L2_OFM_CH, L2_OFM_DIM, 2, 1, L2_SIMD, L2_PE,
 Slice<ap_ufixed<4, 0, AP_TRN, AP_WRAP>>, Slice<ap_uint<4>>, Identity
 >
 (inter2, inter3, weights2, threshs0, numReps, ap_resource_lut());


 TransposeConvLayer_Batch
 <
 L3_K, L3_IFM_CH, L3_IFM_DIM, L3_OFM_CH, L3_OFM_DIM, 2, 1, L3_SIMD, L3_PE,
 Slice<ap_ufixed<4, 0, AP_TRN, AP_WRAP>>, Slice<ap_uint<4>>, Identity
 >
 (inter3, inter4, weights3, threshs0, numReps, ap_resource_lut());


 TransposeConvLayer_Batch
 <
 L4_K, L4_IFM_CH, L4_IFM_DIM, L4_OFM_CH, L4_OFM_DIM, 2, 1, L4_SIMD, L4_PE,
 Slice<ap_ufixed<4, 0, AP_TRN, AP_WRAP>>, Slice<ap_uint<8>>, Identity
 >
 (inter4, inter5, weights4, TanhActivation<ap_fixed<32,16>, ap_int<8>>(), numReps, ap_resource_lut());

 // std::ofstream ofs("/home/uzahid/workspace/QDCGAN/PyTorch/test_image/hls/temp.txt");
 // int size = memOutStream.size();
 // std::cout << "Size of the Stream = " << size << std::endl;
 // int bw = 8;
 // for (int i=0; i<size; i++)
 // {
 //     ap_uint<64> elem = memOutStream.read();
 //     for(int j=0; j<64/bw; j++)
 //     {
 //      ap_int<8> val = elem(bw*(j+1)-1, bw*j);
 //      ap_fixed<8, 2, AP_TRN, AP_WRAP> valc = *reinterpret_cast<ap_fixed<8, 2, AP_TRN, AP_WRAP>*>(&val);
 //      std::cout << std::fixed;
 //      std::cout << std::setprecision(8);
 //      // std::cout << valc << '\n';
 //      ofs << valc << '\n';
 //     }
 // }
 // ofs.close();
 // exit(0);
 
  StreamingDataWidthConverter_Batch<192, 64,  outBits/ 192>(inter5, memOutStream, numReps);
  Stream2Mem_Batch<64, outBits/8>(memOutStream, out, numReps);
}

void BlackBoxJam(ap_uint<64> *in, ap_uint<64> *out, bool doInit,
		unsigned int targetLayer, unsigned int targetMem,
		unsigned int targetInd, unsigned int targetThresh, ap_uint<64> val, unsigned int numReps) {
// pragmas for MLBP jam interface
// signals to be mapped to the AXI Lite slave port
#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS INTERFACE s_axilite port=doInit bundle=control
#pragma HLS INTERFACE s_axilite port=targetLayer bundle=control
#pragma HLS INTERFACE s_axilite port=targetMem bundle=control
#pragma HLS INTERFACE s_axilite port=targetInd bundle=control
#pragma HLS INTERFACE s_axilite port=targetThresh bundle=control
#pragma HLS INTERFACE s_axilite port=val bundle=control
#pragma HLS INTERFACE s_axilite port=numReps bundle=control
// signals to be mapped to the AXI master port (hostmem)
#pragma HLS INTERFACE m_axi offset=slave port=in bundle=hostmem depth=2
#pragma HLS INTERFACE s_axilite port=in bundle=control
#pragma HLS INTERFACE m_axi offset=slave port=out bundle=hostmem depth=16
#pragma HLS INTERFACE s_axilite port=out bundle=control

// partition PE arrays
#pragma HLS ARRAY_PARTITION variable=weights0.m_weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=weights1.m_weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=weights2.m_weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=weights3.m_weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=weights4.m_weights complete dim=1

#pragma HLS ARRAY_PARTITION variable=threshs0.m_thresholds complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs0.m_thresholds complete dim=3

  if (doInit) {
    DoMemInit(targetLayer, targetMem, targetInd, targetThresh, val);
  } else {
    DoCompute(in, out, numReps);
  }

}
