import numpy as np 
import torch as tr
np.random.seed(0)

def new_sw(img, wei, k, IFMDim, STRIDE, PADDING, IFMChannels, SIMD):
    img = img.transpose(0,2,3,1)
    img = img.reshape(-1)
    wei = wei.transpose(1,2,3,0)    
    wei = tr.from_numpy(wei).flip(1).flip(2)
    wei = wei.reshape(-1, 1)

    PADDING = k-1-PADDING
    if PADDING < 0:
        raise Exception("PADDING should be <= K-1")
    STRIDE = STRIDE-1
    if STRIDE < 0:
        raise Exception("Stride should be >= 1")

    IFMDim = IFMDim + (IFMDim-1)*STRIDE
    IFMPadDim = IFMDim + 2*PADDING
    OFMDim = IFMPadDim - k + 1
    IFMLoopBound = IFMDim+PADDING
    # print(f"Padding = {PADDING}, Stride = {STRIDE}, IFMPadDim = {IFMPadDim} OFMDim = {OFMDim}")
    # print(f"IFMLoopBound = {IFMLoopBound}, IFMDim = {IFMDim}")

    multiplying_factor = IFMChannels//SIMD
    number_blocks = k+1
    input_buf = np.empty((number_blocks, IFMPadDim*multiplying_factor, SIMD), dtype=img.dtype)
    cycles_write_block = OFMDim*k**2*multiplying_factor
    cycles_read_block = IFMPadDim*multiplying_factor
    max_cycles = max(cycles_read_block, cycles_write_block)
    baseIter = IFMPadDim*k*multiplying_factor + OFMDim*max_cycles
    counter_internal_block, current_block_write, current_line, read_block = 0, 0, 0, 0
    inp, ofm_y, ofm_x, ky, kx, count_simd = 0, 0, 0, 0, 0, 0
    
    py_img_counter = 0
    inp_i, inp_j = -1*PADDING, -1*PADDING*multiplying_factor
    expand_x, expand_y = 0, 0
    out_list = []
    channel_iter = 0


    for i in range(baseIter):
        if inp < IFMPadDim*k*multiplying_factor:
            for j in range(SIMD):
                if (inp_i < 0 or inp_j < 0 or inp_i >= IFMDim or inp_j >= IFMDim*multiplying_factor):
                    inElem = 0
                elif (expand_x > 0 or expand_y > 0):
                    inElem = 0
                    if j == SIMD-1:
                        expand_x -= 1
                        expand_y -= 1
                else:
                    inElem = img[py_img_counter]
                    py_img_counter += 1
                    if j == SIMD-1:
                        expand_y -= 1
                        if channel_iter == multiplying_factor -1:
                            expand_x = STRIDE*multiplying_factor
                input_buf[current_block_write][current_line][j] = inElem

            channel_iter += 1
            if channel_iter == multiplying_factor:
                channel_iter = 0
            
            inp_j += 1
            if (inp_j == IFMLoopBound*multiplying_factor):
                inp_j = -1*PADDING*multiplying_factor
                inp_i += 1
                if (inp_i == IFMLoopBound):
                    inp_i = -1*PADDING
                if (expand_y < 0):
                    expand_y = IFMDim*STRIDE*multiplying_factor


            inp += 1
            current_line += 1
            if current_line == IFMPadDim*multiplying_factor:
                current_line = 0
                current_block_write += 1
                read_block += 1
                counter_internal_block = 0
                if current_block_write == number_blocks:
                    current_block_write = 0

        else:
            if counter_internal_block < cycles_write_block - 1:
                current_block_read = (current_block_write + 1 + ky)
                if current_block_read >= number_blocks:
                    current_block_read -= number_blocks
                current_line_in_block = (ofm_x+kx)*multiplying_factor + count_simd
                for j in range(SIMD):
                    out_list.append(input_buf[current_block_read][current_line_in_block][j])
                count_simd += 1
                if count_simd == multiplying_factor:
                    count_simd = 0
                    kx+=1
                    if (kx == k):
                        kx = 0
                        ky+=1
                        if (ky==k):
                            ky = 0
                            ofm_x += 1
                            if (ofm_x == OFMDim):
                                ofm_x = 0
                                ofm_y+=1
                                if (ofm_y == OFMDim):
                                    ofm_y = 0
                                    inp = 0
            if counter_internal_block < cycles_read_block-1 and read_block < IFMPadDim:
                for j in range(SIMD):
                    if (inp_i < 0 or inp_j < 0 or inp_i >= IFMDim or inp_j >= IFMDim*multiplying_factor):
                        inElem = 0
                    elif (expand_x > 0 or expand_y > 0):
                        inElem = 0
                        if j == SIMD-1:
                            expand_x -= 1
                            expand_y -= 1
                    else:
                        inElem = img[py_img_counter]
                        py_img_counter += 1
                        if j == SIMD-1:
                            expand_y -= 1
                            if channel_iter == multiplying_factor -1:
                                expand_x = STRIDE*multiplying_factor
                    input_buf[current_block_write][current_line][j] = inElem

                channel_iter += 1
                if channel_iter == multiplying_factor:
                    channel_iter = 0
                
                inp_j += 1
                if (inp_j == IFMLoopBound*multiplying_factor):
                    inp_j = -1*PADDING*multiplying_factor
                    inp_i += 1
                    if (inp_i == IFMLoopBound):
                        inp_i = -1*PADDING
                    if (expand_y < 0):
                        expand_y = IFMDim*STRIDE*multiplying_factor

                current_line += 1
                if current_line == IFMPadDim*multiplying_factor:
                    current_line = 0
                    read_block += 1
                    current_block_write += 1
                    if current_block_write == number_blocks:
                        current_block_write = 0         
            counter_internal_block += 1
            if counter_internal_block == max_cycles -1:
                counter_internal_block = 0
    
    inp_i, inp_j = -1*PADDING, -1*PADDING*multiplying_factor
    expand_x, expand_y = 0, 0    

    read_block = 0
    out_list = np.array(out_list).reshape(-1, IFMChannels*k**2)
    out_list = tr.from_numpy(out_list)
    out = tr.matmul(out_list.type(tr.float64), wei).reshape(1, 1, OFMDim, OFMDim)
    return out


def torch_conv(img, wei, STRIDE, PADDING):
    img_torch = tr.from_numpy(img)
    wei_torch = tr.from_numpy(wei)
    out_conv_torch = tr.nn.functional.conv_transpose2d(img_torch, wei_torch, stride=STRIDE, padding=PADDING)
    return out_conv_torch


if __name__ == "__main__":
    IFMDim = 4
    k = 4
    STRIDE = 2
    PADDING = 1
    IFMChannels = 1
    SIMD = 1

    img = np.random.uniform(-1, 1, size=(1, IFMChannels, IFMDim, IFMDim)).astype(np.float64)
    wei = np.random.uniform(-1, 1, size=(IFMChannels, 1, k, k)).astype(np.float64)

    out_pt = torch_conv(img, wei, STRIDE, PADDING)
    out = new_sw(img, wei, k, IFMDim, STRIDE, PADDING, IFMChannels, SIMD)
    print(tr.allclose(out, out_pt))

































