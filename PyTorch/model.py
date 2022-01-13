import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import brevitas.nn as qnn

from common import *

class DCGAN_gen(nn.Module):
    def __init__(self, out_channels, config):
        super(DCGAN_gen, self).__init__()
        self.config = config
        self.input_size = config.gen_ch
        self.io_bit_width = config.io_bitwidth
        self.act_bit_width = config.activation_bitwidth
        self.weight_bit_width = config.weight_bitwidth
        max_val = 1-1/2**7
        in_ch = config.gen_ch
        out_ch = config.size*8
        layers = []
        num_layers = 5 if config.dataset == "celebA" else 4
        
        for i in range(num_layers):
            (stride, padding) = (1,0) if i == 0 else (2,1)
            layers += [qnn.QuantConvTranspose2d(in_channels=in_ch, out_channels=out_ch, 
                       kernel_size=4, stride=stride, padding=padding, bias=False,
                       weight_bit_width=self.weight_bit_width, weight_quant_type=get_quant_type(self.weight_bit_width),
                       weight_scaling_const=1.0, weight_scaling_impl_type=get_scaling_impl_type(),
                       weight_restrict_scaling_type=get_res_scaling_type(self.weight_bit_width)),]
            if i == (num_layers-1):
                layers += [nn.Tanh()]
            else:
                layers += [qnn.QuantReLU(bit_width=self.act_bit_width, quant_type=get_quant_type(self.act_bit_width), 
                           max_val=1.0, restrict_scaling_type=get_res_scaling_type(self.act_bit_width), 
                           scaling_impl_type=get_scaling_impl_type()),]
            
            in_ch = out_ch
            out_ch = out_channels if i == (num_layers-2) else out_ch//2

        self.model = nn.Sequential(*layers)
        # print(self.model)
        
    # special weightts initialization for GANs
    def init_weights(self, mean, std):
        for mod in self.model:
            normal_init(mod, mean, std)

    # clipping weights of the network
    def clip_weights(self, min_val, max_val):
        for mod in self.model:
            if isinstance(mod, qnn.QuantConvTranspose2d):
                mod.weight.data.clamp_(min_val, max_val)


    def forward(self, x):
        x = x.view(-1, self.input_size, 1, 1)
        # quantizing input to 8 bits
        if self.io_bit_width == 8:
            x = self.quantize(x)
        out = self.model(x)
        return out

    # function to quantize input to Q1.7 format (8-bit) 
    def quantize(self, x, min_val=-1, max_val=0.9921875, frac=7):
        x.clamp_(min_val, max_val)
        x = x*2**frac
        x = torch.round(x)
        x = x/2**frac
        return x

    # exporting the pytorch weights as numpy arrays
    def export(self, path):
        dic = {}
        i = 0

        for mod in self.model:
            if isinstance(mod, qnn.QuantConvTranspose2d):
                weight = mod.int_weight().detach().cpu()
                scale = mod.quant_weight_scale().detach().cpu()
                weight = weight.type(torch.float64)*scale 
                weight = weight.permute(1, 0, 2, 3).flip(2).flip(3).numpy().astype(np.float64)
                dic["arr_"+str(i)] = weight
                i += 1
                if mod.bias is not None:
                    dic["arr_"+str(i)] = mod.bias.detach().numpy().astype(np.float64)
                    i += 1
            elif isinstance(mod, nn.BatchNorm2d):
                dic["arr_"+str(i)] = mod.bias.detach().numpy().astype(np.float64)
                i += 1
                dic["arr_"+str(i)] = mod.weight.detach().numpy().astype(np.float64)
                i += 1
                dic["arr_"+str(i)] = mod.running_mean.detach().numpy().astype(np.float64)
                i += 1
                dic["arr_"+str(i)] = 1./np.sqrt(mod.running_var.detach().numpy().astype(np.float64)+1e-5)
                i += 1
        save_file = path + '/{}-W{}A{}.npz'.format(self.config.dataset, self.weight_bit_width, self.act_bit_width)
        np.savez(save_file, **dic)
        print("Model exported at: ", save_file)


class DCGAN_disc(nn.Module):
    def __init__(self, in_ch, config):
        super(DCGAN_disc, self).__init__()

        layers = []
        num_layers = 5
        out_ch = config.size
        for i in range(num_layers):
            (stride, padding) = (1,0) if i == (num_layers-1) and config.dataset == "celebA" else (2,1)
            
            layers += [nn.Conv2d(in_ch, out_ch, 4, stride, padding, bias=False),]
            layers += [nn.Flatten()] if i == (num_layers-1) else [nn.LeakyReLU(0.2)]

            in_ch = out_ch
            out_ch = 1 if i == (num_layers-2) else out_ch*2

        self.model = nn.Sequential(*layers)

    def clip_weights(self, min_val, max_val):
        for mod in self.model:
            if isinstance(mod, nn.Conv2d):
                mod.weight.data.clamp_(min_val, max_val)


    def init_weights(self, mean, std):
        for mod in self.model:
            normal_init(mod, mean, std)


    def forward(self, x):
        out = self.model(x)
        return out


class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(self,
                 output_blocks=[DEFAULT_BLOCK_INDEX],
                 resize_input=True,
                 normalize_input=True,
                 requires_grad=False):
        
        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, \
            'Last possible output block index is 3'

        self.blocks = nn.ModuleList()

        
        inception = models.inception_v3(pretrained=True)

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps
        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)
        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:
            x = F.interpolate(x,
                              size=(299, 299),
                              mode='bilinear',
                              align_corners=False)

        if self.normalize_input:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp
