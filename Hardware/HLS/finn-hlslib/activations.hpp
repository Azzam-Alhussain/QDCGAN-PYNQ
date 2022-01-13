/******************************************************************************
 *  Copyright (c) 2019, Xilinx, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1.  Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2.  Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *
 *  3.  Neither the name of the copyright holder nor the names of its
 *      contributors may be used to endorse or promote products derived from
 *      this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 *  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 *  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 *  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 *  OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 *  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 *  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 *  ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *******************************************************************************/

/*******************************************************************************
 *
 *  Authors: Giulio Gambardella <giuliog@xilinx.com>
 *           Thomas B. Preusser <thomas.preusser@utexas.edu>
 *             Marie-Curie Fellow, Xilinx Ireland, Grant Agreement No. 751339
 *           Christoph Doehring <cdoehrin@xilinx.com>
 *
 *  @file activations.hpp
 *
 *  Library of templated HLS classes for BNN deployment. 
 *  This file lists a set of classes used to implement  
 *  threshold memory in neural network. 
 *
 *  This project has received funding from the European Union's Framework
 *  Programme for Research and Innovation Horizon 2020 (2014-2020) under
 *  the Marie Skłodowska-Curie Grant Agreement No. 751339.
 *
 *******************************************************************************/

#ifndef ACTIVATIONS_HPP
#define ACTIVATIONS_HPP

#include "interpret.hpp"

#define NUMBER_OF_LUT_ENTRIES 256

static ap_fixed<8 ,2, AP_RND, AP_SAT> lut_tanh[NUMBER_OF_LUT_ENTRIES] = {
-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-0.984375,-0.984375,-0.984375,-0.984375,-0.984375,-0.984375,-0.984375,-0.984375,-0.984375,-0.984375,-0.984375,-0.984375,-0.984375,-0.984375,-0.984375,
-0.984375,-0.984375,-0.984375,-0.984375,-0.984375,-0.984375,-0.984375,-0.984375,-0.984375,-0.96875,-0.96875,-0.96875,-0.96875,-0.96875,-0.96875,-0.96875,-0.96875,-0.96875,-0.96875,
-0.96875,-0.953125,-0.953125,-0.953125,-0.953125,-0.953125,-0.953125,-0.953125,-0.9375,-0.9375,-0.9375,-0.9375,-0.9375,-0.9375,-0.921875,-0.921875,-0.921875,-0.921875,-0.90625,-0.90625,
-0.90625,-0.90625,-0.890625,-0.890625,-0.890625,-0.875,-0.875,-0.875,-0.859375,-0.859375,-0.84375,-0.84375,-0.84375,-0.828125,-0.828125,-0.8125,-0.8125,-0.796875,-0.796875,-0.78125,
-0.765625,-0.765625,-0.75,-0.734375,-0.734375,-0.71875,-0.703125,-0.703125,-0.6875,-0.671875,-0.65625,-0.640625,-0.625,-0.609375,-0.59375,-0.578125,-0.5625,-0.546875,-0.53125,-0.515625,
-0.5,-0.484375,-0.46875,-0.453125,-0.421875,-0.40625,-0.390625,-0.375,-0.34375,-0.328125,-0.3125,-0.28125,-0.265625,-0.234375,-0.21875,-0.203125,-0.171875,-0.15625,-0.125,-0.109375,
-0.078125,-0.0625,-0.03125,-0.015625,0.015625,0.03125,0.0625,0.078125,0.109375,0.125,0.15625,0.171875,0.203125,0.21875,0.234375,0.265625,0.28125,0.3125,0.328125,0.34375,0.375,0.390625,
0.40625,0.421875,0.453125,0.46875,0.484375,0.5,0.515625,0.53125,0.546875,0.5625,0.578125,0.59375,0.609375,0.625,0.640625,0.65625,0.671875,0.6875,0.703125,0.703125,0.71875,0.734375,
0.734375,0.75,0.765625,0.765625,0.78125,0.796875,0.796875,0.8125,0.8125,0.828125,0.828125,0.84375,0.84375,0.84375,0.859375,0.859375,0.875,0.875,0.875,0.890625,0.890625,0.890625,0.90625,
0.90625,0.90625,0.90625,0.921875,0.921875,0.921875,0.921875,0.9375,0.9375,0.9375,0.9375,0.9375,0.9375,0.953125,0.953125,0.953125,0.953125,0.953125,0.953125,0.953125,0.96875,0.96875,
0.96875,0.96875,0.96875,0.96875,0.96875,0.96875,0.96875,0.96875,0.96875,0.984375,0.984375,0.984375,0.984375,0.984375,0.984375,0.984375,0.984375,0.984375,0.984375,0.984375,0.984375,
0.984375,0.984375,0.984375,0.984375,0.984375,0.984375,0.984375,0.984375,0.984375,0.984375,0.984375,0.984375,1,1,1,1,1,1,1,1,1,1};



namespace comp{

  template<typename input_type = void>
    struct greater;

  template<typename input_type = void>
    struct less;

  template<typename input_type = void>
    struct greater_equal;

  template<typename input_type = void>
    struct less_equal;	

  template<typename input_type>
    struct greater : public binary_function<input_type, input_type, ap_uint<1>> {
      ap_uint<1>
      operator()(const input_type& a, const input_type& b) const
      { return a > b; }
    };

  template<typename input_type>
    struct less : public binary_function<input_type, input_type, ap_uint<1>> {
      ap_uint<1>
      operator()(const input_type& a, const input_type& b) const
      { return a < b; }
    };

  template<typename input_type>
    struct greater_equal : public binary_function<input_type, input_type, ap_uint<1>> {
      ap_uint<1>
      operator()(const input_type& a, const input_type& b) const
      { return a >= b; }
    };

  template<typename input_type>
    struct less_equal : public binary_function<input_type, input_type, ap_uint<1>> {
      ap_uint<1>
      operator()(const input_type& a, const input_type& b) const
      { return a <= b; }
    };
	
}

/**
 * General contract for activation functions.
 *
 * This class itself has no formal significance for the implementation
 * of the MVAU. Implementations of activation functions are encouraged
 * to implement it nonetheless to guarantee appropriate function
 * signatures.
 */
template<typename TA, typename TO>
class Activation {
public:
  TA init(unsigned const  nf, unsigned const  pe) const {
#pragma HLS inline
    return  TA(0);
  }

  /**
   * Compute the activation of the passed accumulator value accu in row idx.
   */
  TO activate(unsigned const  nf, unsigned const  pe, TA const &accu) const;
};

/**
 * A no-op activation that simply outputs the computed accumulator
 * output as the final result.
 */
template<typename T>
class PassThroughActivation : public Activation<T, T> {
public:
  T activate(unsigned const  nf, unsigned const  pe, T const &accu) const {
#pragma HLS inline
    return  accu;
  }
};

template<typename T, unsigned factor>
class PassThroughActivation_shift : public Activation<T, T> {
public:
  T activate(unsigned const  nf, unsigned const  pe, T const &accu) const {
#pragma HLS inline
    return  accu*factor;
  }
};

template<typename TA, typename TR>
class TanhActivation : public Activation<TA, TR> {
public:

  const ap_int<8> lower_limit = -3.0;
  const ap_int<8> upper_limit = 3.0;
  const ap_ufixed<8, 6> recip_step = 42.5;

  TR activate(unsigned const  nf, unsigned const  pe, TA const &accu) const {
#pragma HLS inline

    TA input_temp = accu;
    ap_fixed<8,2, AP_RND, AP_SAT> output;

    // If we are outside of LUT range
    if (input_temp <= lower_limit)
    {
      output = lut_tanh[0];
    }
    else if (input_temp >= upper_limit)
    {
      output = lut_tanh[NUMBER_OF_LUT_ENTRIES-1];
    }
    else
    {
      // Scale from [lower, upper] to [0, N]
      TA t = input_temp - lower_limit;
      ap_uint<8> index = t * recip_step;
      output = lut_tanh[index];
    }

    TR out_cast = output*64;
    return out_cast;
  }
};



/**
 * Use a simple global threshold comparison as activation function.
 *
 * The constant threshold is initialized at construction.
 * The default comparison returns true if the threshold value is
 * smaller than the passed accumulator value.
 */
template<typename TA, typename Compare = comp::less<TA>>
class ThresholdActivation : public Activation<TA, bool> {
  TA const  m_threshold;
public:
  ThresholdActivation(TA const &threshold) : m_threshold(threshold) {
#pragma HLS inline
  }

public:
  bool activate(unsigned const  nf, unsigned const  pe, TA const &accu) const {
#pragma HLS inline
    return  Compare()(m_threshold, accu);
  }
};

/*!
 * Use a simple per-row threshold comparison as activation function.
 *
 * The thresholds are taken from an array indexed by output row.
 * It is currently public to allow direct initialization and
 * to make its name accessible for top-level HLS pragmas.
 *
 * The default comparison returns true if the threshold value defined for
 * the indexed row is smaller than the passed accumulator value.
 */
template<unsigned NF, unsigned PE, unsigned NumTH, 
	 typename TA, typename TR, int ActVal = 0, typename Compare = comp::less<TA>>
class ThresholdsActivation {
public:
  TA m_thresholds[PE][NF][NumTH];
  
public:
  TA init(unsigned const  nf, unsigned const  pe) const {
#pragma HLS inline
    return  TA(0);
  }

public:
  TR activate(unsigned const  nf, unsigned const  pe,  TA const &accu) const {
#pragma HLS inline
    TR result=ActVal;
	for(unsigned int i=0; i< NumTH; i++){
#pragma HLS unroll
      result+=Compare()(m_thresholds[pe][nf][i], accu);
    }
    return result;
  }
};


/*!
 * Use a simple per-row threshold comparison as activation function.
 *
 * The thresholds are taken from an array indexed by output row.
 * It is currently public to allow direct initialization and
 * to make its name accessible for top-level HLS pragmas.
 *
 * The default comparison returns true if the threshold value defined for
 * the indexed row is smaller than the passed accumulator value.
 */
template<unsigned NF, unsigned PE, unsigned NumTH, 
   typename TA, typename TR, int ActVal = 0, typename Compare = comp::less<TA>>
class ReLUActivation {
public:
  TA m_thresholds[1][1][NumTH];
  
public:
  TA init(unsigned const  nf, unsigned const  pe) const {
#pragma HLS inline
    return  TA(0);
  }

public:
  TR activate(unsigned const  nf, unsigned const  pe,  TA const &accu) const {
#pragma HLS inline
    TR result=ActVal;
  for(unsigned int i=0; i < NumTH; i++){
#pragma HLS unroll
      result+=Compare()(m_thresholds[0][0][i], accu);
    }
    return result;
  }
};


/*!
 * \brief Use a simple activation function with per-row parameters.
 *
 * The parameters are taken from an array indexed by output row.
 * It is currently public to allow direct initialization and
 * to make its name accessible for top-level HLS pragmas.
 * 
 * \tparam NF    First dimension of the parameter matrix
 * \tparam PE    Second dimension of the parameter matrix
 * \tparam TI    DataType of input layer values
 * \tparam TP    DataType of parameters
 * \tparam TR    DataType of return values
 * \tparam Fxn   Function to be applied on the channel input value
 */

template<unsigned NF, unsigned PE,
   typename TI, typename TP, typename TR, typename Fxn = std::multiplies<TR>>
class ChannelWiseOperation {
public:
  TP parameters[PE][NF];
public:
  TI init(unsigned const  nf, unsigned const  pe) const {
#pragma HLS inline
    return  TI(0);
  }
public:
  TR activate(unsigned const  nf, unsigned const  pe,  TI const &in) const {
#pragma HLS inline
    TR result = Fxn()(parameters[pe][nf], in);
    return result;
  }
};

/*!
 * \brief Thresholding function for multiple images
 *
 * The function performs thresholds comparison with input activation vector, 
 * and generating output based on the comparison results
 *
 * \tparam ImgDimH        Heigth of the Input Feature Map
 * \tparam ImgDimW        Width of the Input Feature Map
 * \tparam NumChannels    Heigth of the input matrix
 * \tparam PE             Number of output rows computed in parallel
 * \tparam TSrcI          DataType of the input activation (as used in the MAC)
 * \tparam TDstI          DataType of the output activation (as generated by the activation)
 * \tparam TI             DataType of the input stream - safely deducible from the paramaters
 * \tparam TO             DataType of the output stream - safely deducible from the paramaters
 * \tparam TA             DataType of the activation class (e.g. thresholds) - safely deducible from the paramaters
 *
 * \param in              Input stream
 * \param out             Output stream
 * \param activation      Activation class
 * \param reps            Number of time the function has to be repeatedly executed (e.g. number of images)
 */
template <
    unsigned ImgDimH, unsigned ImgDimW, unsigned NumChannels, unsigned PE,
    typename TSrcI = Identity, typename TDstI = Identity,
    typename TI, typename TO, typename TA>
void Thresholding_Batch(hls::stream<TI> &in,
                        hls::stream<TO> &out,
                        TA const &activation,
                        int const reps)
{

  // how many different rows each neuron will compute
  // alternatively: number of vertical matrix chunks
  unsigned const NF = NumChannels / PE;

  unsigned nf = 0;
  unsigned tile = 0; // invariant: tile = nf*SF + sf

  // everything merged into a common iteration space (one "big" loop instead
  // of smaller nested loops) to get the pipelinening the way we want
  for (unsigned i = 0; i < reps * ImgDimH * ImgDimW * NF; i++)
  {
    #pragma HLS PIPELINE II=1

    TI inElem;
    inElem = in.read();
    auto outElem = TDstI().template operator()<TO>();
    for (unsigned pe = 0; pe < PE; pe++)
    {
#pragma HLS UNROLL
      auto const act = TSrcI()(inElem);
      outElem(pe,0,1) = activation.activate(nf, pe, act(pe,0));
    }
    out.write(outElem);
    if (++nf == NF)
    {
      nf = 0;
    }
  }
}

/*!
 * \brief Thresholding function for multiple images, with streaming thresholds
 *
 * The function performs thresholds comparison with input activation vector, 
 * and generating output based on the comparison results
 *
 * \tparam ImgDimH        Heigth of the Input Feature Map
 * \tparam ImgDimW        Width of the Input Feature Map
 * \tparam NumChannels    Heigth of the input matrix
 * \tparam PE             Number of output rows computed in parallel
 * \tparam TSrcI          DataType of the input activation (as used in the MAC)
 * \tparam TDstI          DataType of the output activation (as generated by the activation)
 * \tparam ActVal         Initial value of activation at start of thresholding procedure
 * \tparam TT             DataType of the thresholds stream
 * \tparam NumSteps       Number of thresholds per activation
 * \tparam TI             DataType of the input stream - safely deducible from the paramaters
 * \tparam TO             DataType of the output stream - safely deducible from the paramaters
 *
 * \param in              Input stream
 * \param out             Output stream
 * \param weight          Weight stream
 * \param reps            Number of time the function has to be repeatedly executed (e.g. number of images)
 */
template <
    unsigned ImgDimH, unsigned ImgDimW, unsigned NumChannels, unsigned PE,
    typename TSrcI = Identity, typename TDstI = Identity,
    int ActVal=0, typename TT, unsigned int NumSteps,
    typename TI, typename TO>
void Thresholding_Stream_Batch(hls::stream<TI> &in,
                        hls::stream<TO> &out,
                        hls::stream<ap_uint<PE*NumSteps*TT::width>> &weight,
                        int const reps)
{

  // how many different rows each neuron will compute
  // alternatively: number of vertical matrix chunks
  unsigned const NF = NumChannels / PE;

  ThresholdsActivation<1, PE, NumSteps, TT, TO, ActVal, comp::less_equal<TT>> internal_thr;
  #pragma HLS ARRAY_PARTITION variable=internal_thr.m_thresholds complete dim=0

  // everything merged into a common iteration space (one "big" loop instead
  // of smaller nested loops) to get the pipelinening the way we want
  for (unsigned i = 0; i < reps * ImgDimH * ImgDimW * NF; i++)
  {
    #pragma HLS PIPELINE II=1

    ap_uint<PE*NumSteps*TT::width> packed_thr;
    packed_thr = weight.read();
    // slicer to get 1 PE's worth of thresholds
    auto const pe_slicer = Slice<ap_uint<NumSteps*TT::width>>()(packed_thr);

    TI inElem;
    inElem = in.read();
    auto outElem = TDstI().template operator()<TO>();

    for (unsigned pe = 0; pe < PE; pe++)
    {
#pragma HLS UNROLL
      // slicer to get individual thresholds
      auto const thr_slicer = Slice<TT>()(pe_slicer(pe, 0));
      for (unsigned nt = 0; nt < NumSteps; nt++)
      {
      #pragma HLS UNROLL
        internal_thr.m_thresholds[pe][0][nt] = thr_slicer(nt, 0);
      }

      auto const act = TSrcI()(inElem);
      outElem(pe,0,1) = internal_thr.activate(0, pe, act(pe,0));
    }
    out.write(outElem);
  }
}


#endif
