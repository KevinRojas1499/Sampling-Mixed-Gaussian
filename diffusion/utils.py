from datasets import load_dataset
import torch
from pytorch_fid.inception import InceptionV3
import numpy as np
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm

from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
    ToPILImage,
)
from scipy import linalg
from diffusers import DDPMPipeline, DDPMScheduler
import yaml
from models.hf_fno_unet import UNet2DModel
from diffusers import DiffusionPipeline
from datasets import load_dataset, Dataset


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        return config

def load_local_pipeline(config_path, model_path):
    scheduler = noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")
    model_config = load_config(config_path)
    model = UNet2DModel(**model_config)
    model.load_state_dict(torch.load(model_path))
    pipeline = DDPMPipeline(
                    unet=model,
                    scheduler=noise_scheduler,
                )
    return pipeline

def sample_pipeline():
    scheduler = noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")
    
    model_config = load_config("configs/model_configs/default_config.yaml")
    model = UNet2DModel(**model_config)
    model.load_state_dict(torch.load("working/fno-full-ddpm-ema-flowers-64/unet/diffusion_pytorch_model.bin"))
    model.cuda()

    fno_pipeline = DDPMPipeline(
                    unet=model,
                    scheduler=noise_scheduler,
                )
    fno_pipeline.to("cuda")

    generator = torch.Generator(device=fno_pipeline.device).manual_seed(0)
    # run pipeline in inference (sample random noise and denoise)
    data = None
    num_samples = 2048
    batch_size = 16
    for i in range((num_samples + batch_size - 1) // batch_size):
        print("Batch {} of {}".format(i, (num_samples + batch_size - 1) // batch_size))
        images = fno_pipeline(
            generator=generator,
            batch_size=batch_size,
            output_type="numpy",
        ).images
        data = images if data is None else np.concatenate((data, images))
    data = torch.tensor(data).transpose(2, 3).transpose(1, 2).numpy()
    dataset = Dataset.from_dict({"images": data})
    dataset.push_to_hub("Dahoas/fno-full-flowers")


def upload_pipeline():
    config_path = "configs/model_configs/original_config.yaml"
    model_path = "working/unet-ddpm-ema-flowers-64/unet/diffusion_pytorch_model.bin"
    pipeline = load_local_pipeline(config_path, model_path)
    unet = pipeline.unet
    unet.push_to_hub("Dahoas/unet-score")


if __name__ == "__main__":
    #upload_pipeline()
    sample_pipeline()