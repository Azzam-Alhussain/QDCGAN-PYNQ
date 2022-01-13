## Install deps for the Vivado 2019.x
```
sudo apt-get install libjpeg62
sudo add-apt-repository ppa:linuxuprising/libpng12
sudo apt update
sudo apt install libpng12-0
sudo apt install ocl-icd-opencl-dev
```

## Run the Hardware Synthesis
```bash
./make-hw.sh MNIST-W4A4 ultra96 a
```

## Generated Files

The bitstreams will be generated under `output/Bitstreams/` and will also be copied to `Pynq/Bitstreams/`. Moreover the hardware utilization reports are available under `output/Reports/`
