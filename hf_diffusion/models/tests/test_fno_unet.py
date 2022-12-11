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



if __name__ == "__main__":
    test_spectral_conv()
    #test_2()