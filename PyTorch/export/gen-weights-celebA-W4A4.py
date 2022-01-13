# BSD 3-Clause License
# =======

# Copyright (c) 2020, Xilinx
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#  list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#  contributors may be used to endorse or promote products derived from
#  this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import json
import argparse
from finnthesizer import *

parser = argparse.ArgumentParser(description="Quantized DCGAN Training")
parser.add_argument("--hls_files", action="store_true")

if __name__ == "__main__":
    args = parser.parse_args()

    bnnRoot = "."
    npzFile = bnnRoot + "/celebA-W4A4.npz"
    targetDirBin = bnnRoot + "/celebA-W4A4"
    targetDirHLS = bnnRoot + "/celebA-W4A4/hw"
    
    num_classes = 64*64*3
    conv_layers = 5
    size = 32
    gen_ch = 64
    #topology of convolutional layers (only for config.h defines)
    ifm       = [1,  4,  8, 16, 32]
    ofm       = [4,  8, 16, 32, 64]   
    ifm_ch    = [gen_ch, size*8, size*4, size*2, size]
    ofm_ch    = [size*8, size*4, size*2, size*1,    3]   
    filterDim = [ 4, 4, 4, 4, 4]

    WeightsPrecisions_integer =       [1 , 1 , 1 , 1 , 1]
    WeightsPrecisions_fractional =    [3 , 3 , 3 , 3 , 3]
    
    InputPrecisions_integer =         [1 , 0 , 0 , 0 , 0]
    InputPrecisions_fractional =      [7 , 4 , 4 , 4 , 4]
    
    ActivationPrecisions_integer =    [0 , 0 , 0 , 0 , 8]
    ActivationPrecisions_fractional = [4 , 4 , 4 , 4 , 0]

    #configuration of PE and SIMD counts
    # for ultra96
    peCounts =   [4,  8,  8,  8, 3]
    simdCounts = [4, 16, 16, 16, 8]

    # for zcu104
    # peCounts =   [16, 16, 16, 16,  3]
    # simdCounts = [16, 16, 16, 16, 16]

    json_file = {}
    json_file["layers"] = conv_layers
    json_file["pe"] = peCounts
    Wtiles = []
    Ttiles = []

    if not os.path.exists(targetDirBin):
        os.mkdir(targetDirBin)
    if not os.path.exists(targetDirHLS):
        os.mkdir(targetDirHLS)    

    #read weights
    rHW = BNNWeightReader(npzFile, True)

    config = "/**\n"
    config+= " * Finnthesizer Config-File Generation\n";
    config+= " *\n **/\n\n"
    config+= "#ifndef __LAYER_CONFIG_H_\n#define __LAYER_CONFIG_H_\n\n"

    # process convolutional layers
    for convl in range(0, conv_layers):
        peCount = peCounts[convl]
        simdCount = simdCounts[convl]
        WPrecision_fractional = WeightsPrecisions_fractional[convl]
        APrecision_fractional = ActivationPrecisions_fractional[convl]
        IPrecision_fractional = InputPrecisions_fractional[convl]
        WPrecision_integer = WeightsPrecisions_integer[convl]
        APrecision_integer = ActivationPrecisions_integer[convl]
        IPrecision_integer = InputPrecisions_integer[convl]
        print("Using peCount = %d simdCount = %d for engine %d" % (peCount, simdCount, convl))
        # use fixed point weights for the first layer
        (usePopCount, numThresBits, numThresIntBits) = (False, 36, 22) if convl==0 else (False, 32, 20)
        if convl == conv_layers-1:
            (w,t) = rHW.readConvBNComplex_no_thresholds(WPrecision_fractional, APrecision_fractional, IPrecision_fractional, \
                WPrecision_integer, APrecision_integer, IPrecision_integer, \
                usePopCount=usePopCount, numThresBits=numThresBits, numThresIntBits=numThresIntBits)
            paddedH = padTo(w.shape[0], peCount)
            useThresholds = False        
        else:
            (w,t) = rHW.readConvBNComplex_no_thresholds(WPrecision_fractional, APrecision_fractional, IPrecision_fractional, \
                WPrecision_integer, APrecision_integer, IPrecision_integer, \
                usePopCount=usePopCount, numThresBits=numThresBits, numThresIntBits=numThresIntBits)
            paddedH = padTo(w.shape[0], peCount)
            useThresholds = False       
        # compute the padded width and height
        paddedW = padTo(w.shape[1], simdCount)
        # compute memory needed for weights and thresholds
        neededWMem = (paddedW * paddedH) // (simdCount * peCount)
        neededTMem = paddedH // peCount
        print("Layer %d: %d x %d" % (convl, paddedH, paddedW))
        print("WMem = %d TMem = %d" % (neededWMem, neededTMem))
        print("IPrecision = %d.%d WPrecision = %d.%d APrecision = %d.%d" % (IPrecision_integer, IPrecision_fractional, \
            WPrecision_integer,WPrecision_fractional, APrecision_integer, APrecision_fractional))
        m = BNNProcElemMem(peCount, simdCount, neededWMem, neededTMem, WPrecision_integer, APrecision_integer, IPrecision_integer, \
            WPrecision_fractional, APrecision_fractional, IPrecision_fractional, numThresBits=numThresBits, numThresIntBits=numThresIntBits)
        
        m.addMatrix(w,t,paddedW,paddedH)
        config += (printConvDefines("L%d" % convl, filterDim[convl], ifm_ch[convl], ifm[convl], ofm_ch[convl], ofm[convl], simdCount, \
            peCount, neededWMem, neededTMem, WPrecision_integer, APrecision_integer, WPrecision_fractional, APrecision_fractional)) + "\n" 
        Wtiles.append(neededWMem)
        Ttiles.append(neededTMem)
        if args.hls_files:        
            #generate HLS weight and threshold header file to initialize memory directly on bitstream generation
            m.createHLSInitFiles(targetDirHLS + "/memdata-" + str(convl) + ".h", str(convl), useThresholds=useThresholds)
        else:
            #generate binary weight and threshold files to initialize memory during runtime
            #because HLS might not work for very large header files        
            m.createBinFiles(targetDirBin, str(convl))

    config+="\n#define IMG_DIM %d" %ifm[0]
    config+="\n#define IMG_CH %d" %ifm_ch[0]
    config+="\n#define no_cl %d" %num_classes
    config+="\n#define LL_MH %d" %padTo(num_classes, 64)
    config+="\n#define BAKED_WEIGHTS %d" %args.hls_files    
    config+="\n\n#endif //__LAYER_CONFIG_H_\n\n"
    configFile = open(targetDirHLS+"/config.h", "w")
    configFile.write(config)
    configFile.close()
    json_file["Wtiles"] = Wtiles
    json_file["Ttiles"] = Ttiles
    json_file["BAKED_WEIGHTS"] = args.hls_files
    with open(targetDirHLS+"/config.json", 'w') as outfile:
        json.dump(json_file, outfile)

    if args.hls_files:
        print("Only HLS Files Generated...")
    else:
        print("Only Bin Files Generated...")

