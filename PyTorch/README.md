# Docker
```bash
cd  brevitas-docker
./run-docker.sh
cd QDCGAN/PyTorch/
````

# MNIST 

## Training (float)
```bash
python main.py --gen_ch 16 --size 8 --gpus 0 --dataset MNIST

```

## Quantized Training (WmAn)
```bash
python main.py --gen_ch 16 --size 8 --iobw 8 --wbw 4 --abw 4 --gpus 0 --dataset MNIST --resume <path to checkpoint.tar>
```

# celebA

## Training (float)
```bash
python main.py --gen_ch 64 --size 32 --gpus 0 --dataset celebA
```

## Quantized Training (WmAn)
```bash
python main.py --gen_ch 64 --size 32 --iobw 8 --wbw 4 --abw 4 --gpus 0 --dataset celebA --resume <path to checkpoint.tar>
```

## Evaluate
```bash
python main.py --gen_ch 64 --size 32 --iobw 8 --wbw 4 --abw 4 --gpus 0  --evaluate --examples 5 --dataset celebA --resume <path to checkpoint.tar>
```

## Export
```bash
python main.py --gen_ch 64 --size 32 --iobw 8 --wbw 4 --abw 4 --dataset celebA --export --resume <path to checkpoint.tar> 
```

## Calculate FID:
```bash
python main.py --gen_ch 64 --size 32 --iobw 8 --wbw 4 --abw 4 --gpus 0 --fid --resume <path to checkpoint.tar> 
```

## Generate Bin files for FPGA
```bash
cd export
python gen-weights-celebA-W4A4.py
```


# Running on Jetson
```bash
cd brevitas-docker
./run-docker-jetson.sh
cd QDCGAN/PyTorch/
```

And then use the same Evaluate commands as given above but with `python3` and without the `--gpus 0` flag.
