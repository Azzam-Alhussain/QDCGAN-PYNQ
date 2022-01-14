import os
import pynq
import numpy as np
import time
if os.environ['BOARD'] == 'Ultra96':
    PLATFORM="ultra96"
elif os.environ['BOARD'] == 'ZCU104':
    PLATFORM="zcu104"
    import matplotlib
    # matplotlib.use('Agg')
else:
    raise RuntimeError("Board not supported")
import matplotlib.pyplot as plt
import itertools
import argparse
import cffi
import json

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
PARAMS_PATH = ROOT_DIR + "/../../PyTorch/export/"

_ffi = cffi.FFI()

_ffi.cdef("""
void load_layer(const char* path, unsigned int layer, unsigned int PEs, unsigned int Wtiles, unsigned int Ttiles, unsigned int API, unsigned int addr);
void deinit();
"""
)

_libraries = {}

parser = argparse.ArgumentParser(description="Quantized DCGAN Inference on Pynq")
parser.add_argument("--wbw", dest="weight_bitwidth", type=int, default=4)
parser.add_argument("--abw", dest="activation_bitwidth", type=int, default=4)
parser.add_argument("--dataset", dest="dataset", type=str, default="MNIST")
parser.add_argument("--examples", default=1, type=int)
parser.add_argument("--gen_ch", default=16, type=int)

class hgan:
    def __init__(self, bitstream_path, buffer_size, params_path=None):
        print(f"Loading bitstream: {bitstream_path}")
        self.overlay = pynq.Overlay(bitstream_path, download=True)
        self.bbj = self.overlay.BlackBoxJam_0.register_map
        self.in_buffer = None
        self.out_buffer = None
        self.buffer_size = (buffer_size*8)//64

        # weights loading code
        self.base_addr = self.overlay.BlackBoxJam_0.mmio.base_addr
        dllname = "layer_loader.so"
        if dllname not in _libraries:
            _libraries[dllname] = _ffi.dlopen(os.path.join(ROOT_DIR, dllname))
        self.interface = _libraries[dllname]

        # loading network's config file
        with open(params_path + "/hw/config.json") as json_file:
            self.config = json.load(json_file)

        if params_path is not None and not self.config["BAKED_WEIGHTS"]:
            self.load_parameters(params_path)
        else:
            print("Using baked in weights and thresholds (if any) of the accelerator...")

    def __del__(self):
        self.interface.deinit()

    # function to set weights and activation thresholds of specific network
    def load_parameters(self, params_path):
        if os.path.isdir(params_path):
            start = time.time()
            self.params_loader(params_path)
            end = time.time() - start
            print("Parameter loading took {:.2f} sec...".format(end))
        else:
            print("\nERROR: No such parameter directory \"" + params_path + "\"")


    def params_loader(self, params):
        print("Setting network weights and thresholds in accelerator...")
        for layer in range(self.config["layers"]):
            self.interface.load_layer(params.encode(), layer, self.config["pe"][layer], \
                self.config["Wtiles"][layer], self.config["Ttiles"][layer], 0, self.base_addr)

    def allocate_io_buffers(self, inp, count=1):
        self.in_buffer = pynq.allocate(shape=inp.shape, dtype=np.uint64)
        self.out_buffer = pynq.allocate(shape=(self.buffer_size*count,), dtype=np.uint64)
        np.copyto(self.in_buffer, np.zeros(shape=inp.shape, dtype=np.uint64))
        np.copyto(self.out_buffer, np.zeros(shape=(self.buffer_size*count,), dtype=np.uint64))
        self.bbj.in_V_1 = self.in_buffer.physical_address & 0xffffffff
        self.bbj.in_V_2 = (self.in_buffer.physical_address >> 32) & 0xffffffff
        self.bbj.out_V_1 = self.out_buffer.physical_address & 0xffffffff
        self.bbj.out_V_2 = (self.out_buffer.physical_address >> 32) & 0xffffffff

    def __call__(self, x):
        x_processed = self.preprocess_input(x)
        if self.in_buffer is None:
            self.allocate_io_buffers(x_processed, x.shape[0])
        np.copyto(self.in_buffer, x_processed)
        self.bbj.numReps = x.shape[0]
        self.bbj.doInit = 0

        start = time.time()
        self.start()
        print("FPS = {}".format(x.shape[0]/(time.time()-start)))
        
        pred = np.copy(np.frombuffer(self.out_buffer, dtype=np.uint64))
        pred = self.preprocess_output(pred)
        return pred


    def start(self):
        self.bbj.CTRL.AP_START = 1
        while not self.bbj.CTRL.AP_DONE:
            pass
        
    def preprocess_input(self, x):
        x = np.clip(x, -1, 1)
        x = np.round(x*2**7)
        x = x.astype(np.int8).view(np.uint64)
        return x


    def preprocess_output(self, x):
        x = x.view(np.int8)/2**6
        x = (x+1)/2
        return x

if __name__=='__main__':
    args = parser.parse_args()
    if args.dataset == "MNIST":
        (dim, channel) = (32, 1) 
    elif args.dataset == "celebA":
        (dim, channel) = (64, 3)
    else:
        raise Exception(f"Invalid Dataset: {args.dataset}")
    buffer_size = dim*dim*channel
    PARAMS_PATH += f"{args.dataset}-W{args.weight_bitwidth}A{args.activation_bitwidth}/"
    bitstream_path = f"Bitstreams/{args.dataset}-W{args.weight_bitwidth}A{args.activation_bitwidth}-{PLATFORM}.bit"
    name = f"{args.dataset}-W{args.weight_bitwidth}A{args.activation_bitwidth}.png"
    gan = hgan(bitstream_path, buffer_size, PARAMS_PATH)

    inp = np.random.randn(args.examples*args.examples, args.gen_ch)
    out = gan(inp)
    if args.dataset == "MNIST":
        out = out.reshape(-1, dim, dim)
        cmap = 'gray'
    else:
        out = out.reshape(args.examples*args.examples, dim, dim, channel)
        cmap = None
    if args.examples > 1:
        size_figure_grid = args.examples
        fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(args.examples,args.examples))
        for i,j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
            ax[i,j].get_xaxis().set_visible(False)
            ax[i,j].get_yaxis().set_visible(False)

        for k in range(args.examples*args.examples):
            i = k//args.examples
            j = k%args.examples
            single_image = out[k]                
            if args.dataset != "MNIST":
                single_image = (((single_image - single_image.min()) * 255) / (single_image.max() - single_image.min())).astype(np.uint8)
            ax[i,j].cla()
            ax[i,j].imshow(single_image, cmap=cmap)
        plt.savefig(name)
        plt.show()

    else:
        single_image = out[0]
        if args.dataset != "MNIST":
            single_image = (((single_image - single_image.min()) * 255) / (single_image.max() - single_image.min())).astype(np.uint8)
        else:
            single_image *= 255
        from PIL import Image
        from matplotlib import cm
        im = Image.fromarray(np.uint8(single_image))
        im.save(name)
        plt.imshow(single_image, cmap="gray")
        plt.show()        
