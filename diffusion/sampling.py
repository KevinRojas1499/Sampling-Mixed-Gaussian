from utils import load_local_pipeline
import torch
from tqdm import tqdm


def ddpm_step():
	return None

def fourier_step():
	return None

def fourier_sampling():
	config_path = "configs/model_configs/original_config.yaml"
    model_path = "working/unet-ddpm-ema-flowers-64/unet/diffusion_pytorch_model.bin"
	pipeline = load_local_pipeline(config_path, model_path)

	model = pipeline.unet
	scheduler = pipeline.scheduler

	num_inference_steps = 1000
	scheduler.set_timesteps(num_inference_steps)

	data_shape = (64, 64)
	sample = torch.randn(data_shape).to("cuda")

	for t in tqdm(pipeline.scheduler.timesteps):
		score = model(sample, t).sample
		sample = fourier_step()


if __name__ == "__main__":
	fourier_sampling()