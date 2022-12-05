from fno_block import SpectralConv2d
import torch


def test_1():
	spectral_conv = SpectralConv2d(128, 128, 64 // 2 + 1, 64 // 2 + 1)

	batch = torch.randn((64, 128, 64, 64))
	spectral_conv(batch)

def test_2():
	c_m = torch.randn((2, 2), dtype=torch.cfloat)
	c_v = torch.randn(2, dtype=torch.cfloat)
	print(c_m)
	print(c_v)
	cr_v = torch.view_as_real(c_v)
	print(cr_v)

if __name__ == "__main__":
	#test_1()
	test_2()