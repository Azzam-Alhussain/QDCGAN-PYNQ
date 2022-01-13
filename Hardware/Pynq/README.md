## Install the dependencies
```bash
sudo apt update && sudo apt upgrade -y && sudo apt install -y vnc4server python3-tk
```

## Launch the Vnc
```bash
sudo vncserver -kill :1 # kill the previous vncserver service if any
sudo vncserver -geometry 1366x768
```

## Launch the inference from the Vnc session
```bash
cd /home/xilinx/QDCGAN/Hardware/Pynq
./gen.sh   # to generate the parameters loading driver
python3 main.py --examples 5 --wbw 4 --abw 4 --dataset <MNIST/celebA> --gen_ch <32/64>
```

