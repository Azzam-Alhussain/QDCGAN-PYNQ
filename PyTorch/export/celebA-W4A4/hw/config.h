/**
 * Finnthesizer Config-File Generation
 *
 **/

#ifndef __LAYER_CONFIG_H_
#define __LAYER_CONFIG_H_

/**
 * Convolutional Layer L0:
 *      IFM  =     1  IFM_CH =    64
 *      OFM  =     4  OFM_CH =   256
 *     SIMD  =     4    PE   =     4
 *     WMEM  = 16384   TMEM  =    64
 *     #Ops  = 8388608   Ext Latency  = 262144
**/

#define L0_K 4
#define L0_IFM_CH 64
#define L0_IFM_DIM 1
#define L0_OFM_CH 256
#define L0_OFM_DIM 4
#define L0_SIMD 4
#define L0_PE 4
#define L0_WMEM 16384
#define L0_TMEM 64
#define L0_WPI 1
#define L0_API 0
#define L0_WPF 3
#define L0_APF 4

/**
 * Convolutional Layer L1:
 *      IFM  =     4  IFM_CH =   256
 *      OFM  =     8  OFM_CH =   128
 *     SIMD  =    16    PE   =     8
 *     WMEM  =  4096   TMEM  =    16
 *     #Ops  = 67108864   Ext Latency  = 262144
**/

#define L1_K 4
#define L1_IFM_CH 256
#define L1_IFM_DIM 4
#define L1_OFM_CH 128
#define L1_OFM_DIM 8
#define L1_SIMD 16
#define L1_PE 8
#define L1_WMEM 4096
#define L1_TMEM 16
#define L1_WPI 1
#define L1_API 0
#define L1_WPF 3
#define L1_APF 4

/**
 * Convolutional Layer L2:
 *      IFM  =     8  IFM_CH =   128
 *      OFM  =    16  OFM_CH =    64
 *     SIMD  =    16    PE   =     8
 *     WMEM  =  1024   TMEM  =     8
 *     #Ops  = 67108864   Ext Latency  = 262144
**/

#define L2_K 4
#define L2_IFM_CH 128
#define L2_IFM_DIM 8
#define L2_OFM_CH 64
#define L2_OFM_DIM 16
#define L2_SIMD 16
#define L2_PE 8
#define L2_WMEM 1024
#define L2_TMEM 8
#define L2_WPI 1
#define L2_API 0
#define L2_WPF 3
#define L2_APF 4

/**
 * Convolutional Layer L3:
 *      IFM  =    16  IFM_CH =    64
 *      OFM  =    32  OFM_CH =    32
 *     SIMD  =    16    PE   =     8
 *     WMEM  =   256   TMEM  =     4
 *     #Ops  = 67108864   Ext Latency  = 262144
**/

#define L3_K 4
#define L3_IFM_CH 64
#define L3_IFM_DIM 16
#define L3_OFM_CH 32
#define L3_OFM_DIM 32
#define L3_SIMD 16
#define L3_PE 8
#define L3_WMEM 256
#define L3_TMEM 4
#define L3_WPI 1
#define L3_API 0
#define L3_WPF 3
#define L3_APF 4

/**
 * Convolutional Layer L4:
 *      IFM  =    32  IFM_CH =    32
 *      OFM  =    64  OFM_CH =     3
 *     SIMD  =     8    PE   =     3
 *     WMEM  =    64   TMEM  =     1
 *     #Ops  = 12582912   Ext Latency  = 262144
**/

#define L4_K 4
#define L4_IFM_CH 32
#define L4_IFM_DIM 32
#define L4_OFM_CH 3
#define L4_OFM_DIM 64
#define L4_SIMD 8
#define L4_PE 3
#define L4_WMEM 64
#define L4_TMEM 1
#define L4_WPI 1
#define L4_API 8
#define L4_WPF 3
#define L4_APF 0


#define IMG_DIM 1
#define IMG_CH 64
#define no_cl 12288
#define LL_MH 12288
#define BAKED_WEIGHTS 0

#endif //__LAYER_CONFIG_H_

