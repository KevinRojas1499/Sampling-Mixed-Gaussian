from fno_block import SpectralConv2d
import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
from fno_block import FNODownBlock2D
from block import DownBlock2D


def test_spectral_conv():
    spectral_conv = SpectralConv2d(128, 128, 64 // 2 + 1, 64 // 2 + 1)
    batch = torch.randn((64, 128, 64, 64))
    output = spectral_conv(batch)
    print(output.shape)


def test_2():
    c_m = torch.randn((2, 2), dtype=torch.cfloat)
    c_v = torch.randn(2, dtype=torch.cfloat)
    print(c_m)
    print(c_v)
    cr_v = torch.view_as_real(c_v)
    print(cr_v)


def compare_param_sizes():
    import time
    start = time.time()
    fno_block = FNODownBlock2D(128, 256, 3, num_layers=2, samples=256)
    t = time.time()
    print(t - start)
    block = DownBlock2D(128, 256, 3, num_layers=2, samples=256)
    print(time.time() - t)

    
    print("FNO num params: {}".format(len(list(fno_block.parameters()))))
    [print(p.size()) for p in fno_block.parameters()]
    print("Block num params: {}".format(len(list(block.parameters()))))
    [print(p.size()) for p in block.parameters()]



if __name__ == "__main__":
    #test_spectral_conv()
    compare_param_sizes()
    #test_2()