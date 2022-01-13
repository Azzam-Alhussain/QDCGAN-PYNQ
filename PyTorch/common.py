import numpy as np
import torch
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.core.quant import QuantType
from brevitas.core.scaling import ScalingImplType
from brevitas.core.restrict_val import RestrictValueType

def dump(x, path, fmt='%.8f'):
    print(x.shape)
    if x.dim() == 4:
        x = x.permute(0,2,3,1)
    x = x.detach().cpu().numpy()
    x = x.reshape(-1)
    np.savetxt(path, x, fmt=fmt)
    print('Tensor dumped at {}'.format(path))


def dump_wei(mod, path, fmt='%.8f'):
    weight = mod.int_weight().detach().cpu()
    print(weight.shape)
    scale = mod.quant_weight_scale().detach().cpu()
    weight = weight.type(torch.float64)*scale 
    weight = weight.permute(1, 0, 2, 3).flip(2).flip(3).numpy().astype(np.float64)

    weight = np.moveaxis(weight, 1, -1)
    # weight = weight.reshape((numOut, fanin))
    weight = weight.reshape(-1)
    np.savetxt(path, weight, fmt=fmt)
    print('Tensor dumped at {}'.format(path))


def get_quant_type(bit_width):
    if bit_width == 32:
        return QuantType.FP
    elif bit_width == 1:
        return QuantType.BINARY
    else:
        return QuantType.INT

def get_scaling_impl_type():
	return ScalingImplType.CONST

def get_res_scaling_type(bit_width):
	if 1 < bit_width < 32:
		return RestrictValueType.POWER_OF_TWO
	else:
		return RestrictValueType.LOG_FP


def normal_init(mod, mean, std):
    if isinstance(mod, (qnn.QuantConvTranspose2d, qnn.QuantConv2d, nn.Conv2d, nn.ConvTranspose2d)):
        mod.weight.data.normal_(mean, std)
        if mod.bias is not None:
            mod.bias.data.zero_()
    elif isinstance(mod, nn.BatchNorm2d):
        mod.weight.data.normal_(1.0, 0.02)
        if mod.bias is not None:
            mod.bias.data.zero_()       