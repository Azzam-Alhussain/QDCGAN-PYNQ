# MIT License
#
# Copyright (c) 2019 Xilinx
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import logging
import sys
import os


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class TrainingEpochMeters(object):
    def __init__(self):
        self.disc_loss = AverageMeter()
        self.gen_loss = AverageMeter()


class Logger(object):

    def __init__(self, output_dir_path, dry_run):
        self.output_dir_path = output_dir_path
        self.log = logging.getLogger('log')
        self.log.setLevel(logging.INFO)

        # Stout logging
        out_hdlr = logging.StreamHandler(sys.stdout)
        out_hdlr.setFormatter(logging.Formatter('%(message)s'))
        out_hdlr.setLevel(logging.INFO)
        self.log.addHandler(out_hdlr)

        # Txt logging
        if not dry_run:
            file_hdlr = logging.FileHandler(os.path.join(self.output_dir_path, 'log.txt'))
            file_hdlr.setFormatter(logging.Formatter('%(message)s'))
            file_hdlr.setLevel(logging.INFO)
            self.log.addHandler(file_hdlr)
            self.log.propagate = False

    def info(self, arg):
        self.log.info(arg)

    def training_batch_cli_log(self, epoch_meters, epoch, tot_ep, batch, tot_batches):
        self.info('Epoch: [{0}/{1}][{2}/{3}]\t'
                         'Discriminator Loss {disc_loss.val:.4f} ({disc_loss.avg:.4f})\t'
                         'Generator Loss {gen_loss.val:.4f} ({gen_loss.avg:.4f})\t'
                         .format(epoch, tot_ep, batch, tot_batches,
                                 disc_loss=epoch_meters.disc_loss,
                                 gen_loss=epoch_meters.gen_loss))
