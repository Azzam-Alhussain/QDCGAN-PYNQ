# Quantized Deconvolution GANs on Xilinx SoC-FPGAs
### [Azzam Alhussain](http://azzam.page/), [Mingjie Lin](https://www.ece.ucf.edu/person/mingjie-lin/)
___
**This is the official HW/SW Co-design efficient training and implementation of quantized deconvolution GAN (QDCGAN) on PYNQ FPGAs and Jetson nano frameworks that is accepted and will be published soon as a conference paper in the IEEE Xplore Digital Library as [Hardware-Efficient Deconvolution-Based GAN for Edge Computing](https://ieeexplore.ieee.org/Xplore/home.jsp), and will be presented in March 2022 at the [56th Annual Conference on Information Sciences and Systems (CISS)](https://ee-ciss.princeton.edu/).**

## Description

This paper proposed a HW/SW co-design approach for training quantized deconvolution GAN (QDCGAN) implemented on PYNQ FPGAs using a scalable streaming dataflow architecture capable of achieving higher throughput versus resource utilization trade-off. The developed accelerator is based on an efficient deconvolution engine that offers high parallelism with respect to PE & SIMD scaling factors for GAN-based edge computing. Lastly, MNIST & celebA datasets, and network scalability were analyzed for low-power inference on resource-constrained platforms. 

## Contributions
- Developed a scalable inference accelerator for transpose convolution operation for quantized DCGAN (QDCGAN) on top of [FINN by Xilinx](https://xilinx.github.io/finn/). 
- Provided a complete open-source framework (training to implementation stack) for investigating the effect of variable bit widths for weights and activations. 
- Demonstrated that the weights and activations influence performance measurement, resource utilization, throughput, and the quality of the generated images.
- The community can build upon our code, explore, and search efficient implementation of SRGAN on low-power FPGAs which are considered as a solution for a wide range of medical   and microscopic imaging applications.

## Getting Started

### Requirement
* Nvidia GPU
* Linux Ubuntu 18.04
* Python 3.6+
* Pytorch 1.4.0+
* Vivado 2019.3+ 
* PYNQ framework 2.6
* Xilinx SoC-FPGAs Pynq supported (ex: Ultra96 & ZCU104)

### HW/SW training & implementation

- `PyTorch` folder for training.
- `Hardware` folder for the synthesis of the accelerator.
- `Hardware/Pynq/` folder for deployment on xilinx SOC-FPGAs having pynq linux.

## License

All source code is made available under a BSD 3-clause license. You can freely use and modify the code, without warranty, so long as you provide attribution
to the authors. See `LICENSE.md` for the full license text.

The manuscript text is currently accepted, and will be published soon as a conference paper in the IEEE Xplore Digital Library.

## Citation

Please use the below citation till it updated from IEEE Xplore Digital Library,

**Alhussain, A. and Lin, M, "Hardware-Efficient Deconvolution-Based GAN for Edge Computing," in 56th Annual Conference on Information Sciences and Systems (CISS) 2022, Virtual IEEE conference, March 9-11, 2022. pp.1-5, Accessed on: Jan. 19, 2022. [Online]. Available: https://arxiv.org/abs/2201.06878**

## Acknowledgments

Inspiration, code snippets, references, etc.

* [FINN: A Framework for Fast, Scalable Binarized Neural Network Inference](https://xilinx.github.io/finn/)
* [Xilinx/finn-hlslib](https://github.com/Xilinx/finn-hlslib)
* [Xilinx/brevitas](https://github.com/Xilinx/brevitas)
* [Xilinx/PYNQ](https://github.com/Xilinx/PYNQ)
* [Xilinx/BNN-PYNQ](https://github.com/Xilinx/BNN-PYNQ)
* [A Competitive Edge: Can FPGAs Beat GPUs at DCNN Inference Acceleration in Resource-Limited Edge Computing Applications?](https://arxiv.org/pdf/2102.00294v2.pdf)
