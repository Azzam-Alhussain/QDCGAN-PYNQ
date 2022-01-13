#pragma once
#include <string>
#include <iostream>
#include "ap_int.h"
#include "config.h"
#include "bnn-library.h"
using namespace std;

typedef unsigned long long ExtMemWord;

const unsigned int bitsPerExtMemWord = sizeof(ExtMemWord)*8;

#define INPUT_BUF_ENTRIES     3840000
#define OUTPUT_BUF_ENTRIES    250000
#define FOLDEDMV_INPUT_PADCHAR  0

ExtMemWord * bufIn, * bufOut;

void BlackBoxJam(ap_uint<64> * in, ap_uint<64> * out, bool doInit, unsigned int targetLayer, 
                 unsigned int targetMem, unsigned int targetInd, unsigned int targetThresh, 
                 ap_uint<64> val, unsigned int numReps);

void FoldedMVInit(const char * attachName) {
  if (!bufIn) {
    bufIn = new ExtMemWord[INPUT_BUF_ENTRIES];
    if (!bufIn) {
      cout << "Failed to allocate host buffer" << endl;
      exit(1);
    }
  }
  if (!bufOut) {
    bufOut = new ExtMemWord[OUTPUT_BUF_ENTRIES];
    if (!bufOut) {
      cout << "Failed to allocate host buffer"<< endl;
      exit(1);
    }
  }
}

void FoldedMVDeinit() {
  delete bufIn;
  delete bufOut;
  bufIn = 0;
  bufOut = 0;
}

void FoldedMVMemSet(unsigned int targetLayer, unsigned int targetMem, unsigned int targetInd,unsigned int targetThresh, ExtMemWord val) {
  // call the accelerator in weight init mode
  BlackBoxJam((ap_uint<64> *)bufIn, (ap_uint<64> *)bufOut, true, targetLayer, targetMem, targetInd,targetThresh, val, 0);
}



void FoldedMVLoadLayerMem(std::string dir, unsigned int layerNo, unsigned int peCount, unsigned int linesWMem, unsigned int linesTMem, unsigned int cntThresh) {
  for(unsigned int pe = 0; pe < peCount; pe++) {
    // load weights
    ifstream wf(dir + "/" + to_string(layerNo) + "-" + to_string(pe) + "-weights.bin", ios::binary | ios::in);
    if(!wf.is_open()) {
      cout << "Could not open file"<< endl;
      exit(1);
    }
    for(unsigned int line = 0 ; line < linesWMem; line++) {
      ExtMemWord e = 0;
      wf.read((char *)&e, sizeof(ExtMemWord));
      FoldedMVMemSet(layerNo * 2, pe, line, 0, e);
    }
    wf.close();

    // load thresholds
    if (cntThresh > 0)
    {
      ifstream tf(dir + "/" + to_string(layerNo) + "-" + to_string(pe) + "-thres.bin", ios::binary | ios::in);
      if(!tf.is_open()){
        cout << "Could not open file"<< endl;
        exit(1);
      }
      for(unsigned int line = 0 ; line < linesTMem; line++) {
        for(unsigned int i = 0; i < cntThresh; i++){
          ExtMemWord e = 0;
          tf.read((char *)&e, sizeof(ExtMemWord));
          FoldedMVMemSet(layerNo * 2 + 1, pe, line,i, e);
        }
      }
      tf.close();
    }
  }
}


// return in padded to a multiple of padTo
unsigned int paddedSize(unsigned int in, unsigned int padTo) {
  if(in % padTo == 0) {
    return in;
  } else {
    return in + padTo - (in % padTo);
  }
}

template<unsigned int inWidth, unsigned int SIMDWidth>
void quantiseAndPack_floats(const float in[IMG_CH], ExtMemWord * out, unsigned int inBufSize=INPUT_BUF_ENTRIES) {
  if((IMG_CH * inWidth) > (inBufSize * bitsPerExtMemWord)) {
    cout << "Not enough space in input buffer"<< endl;
    exit(1);
  }
  // first, fill the target buffer with padding data
  memset(out, 0, inBufSize * sizeof(ExtMemWord));
  ExtMemWord tmpv[bitsPerExtMemWord / inWidth];
  // now pack each quantised value as required.
  for(unsigned int i=0; i < IMG_CH; i++) {
    ap_fixed<inWidth, 1, AP_RND, AP_SAT> fxdValue = in[i];
    ap_uint<inWidth> uValue = *reinterpret_cast<ap_uint<inWidth> *>(&fxdValue); // Interpret the fixed value as an integer.
    ExtMemWord v = ((ExtMemWord)uValue & (~(ExtMemWord)0 >> (bitsPerExtMemWord - inWidth))); // Zero all bits except for the (bitsPerExtMemWord - inWidth) least significant bits.
    out[i / (bitsPerExtMemWord / inWidth)] |= (v << inWidth*(i % (bitsPerExtMemWord / inWidth)));
  }
}

template<unsigned int inWidth, unsigned int outWidth, unsigned int count, typename LowPrecType>
LowPrecType* test_gan(float imgs[count*IMG_CH], float &usecPerImage)
{
  cout << "Packing inputs..." << endl;
  // number of ExtMemWords per image
  const unsigned int psi = paddedSize(IMG_CH*inWidth, bitsPerExtMemWord) / bitsPerExtMemWord;
  // number of ExtMemWords per output
  const unsigned int pso = paddedSize(LL_MH*outWidth, bitsPerExtMemWord) / bitsPerExtMemWord;
  if(INPUT_BUF_ENTRIES < count*psi)
  {
    cout << "Not enough space in accelBufIn"<< endl;
    exit(1);
  }
  if(OUTPUT_BUF_ENTRIES < count*pso) {
    cout << "Not enough space in accelBufOut"<< endl;
    exit(1);
  }
  // allocate host-side buffers for packed input and outputs
  ExtMemWord * packedImages = new ExtMemWord[(count * psi)];
  ExtMemWord * packedOut = new ExtMemWord[(count * pso)];

  for(unsigned int i = 0; i < count; i++) {
    quantiseAndPack_floats<inWidth, 1>(imgs, &packedImages[i * psi], psi);
  }
  cout << "Running test for " << count << " inputs..." << endl;

  auto t1 = chrono::high_resolution_clock::now();
  // call the accelerator in compute mode
  BlackBoxJam((ap_uint<64> *)packedImages, (ap_uint<64> *)packedOut, false, 0, 0, 0, 0, 0, count);
  auto t2 = chrono::high_resolution_clock::now();
  
  LowPrecType * result = (LowPrecType*) packedOut;

  auto duration = chrono::duration_cast<chrono::microseconds>( t2 - t1 ).count();
  usecPerImage = (float)duration / (count);
  cout << "Inference took " << duration << " microseconds, " << usecPerImage << " usec per image" << endl;

  return result;

}
