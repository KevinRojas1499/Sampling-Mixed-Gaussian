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

def sample_pipeline(rank=0, world_size=1, load_model=True, num_samples=50000, result_queue=None):
    if world_size > 1:
        import torch.distributed as dist
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

    if load_model:
        scheduler = noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")
    
        model_config = load_config("configs/model_configs/default_config.yaml")
        model = UNet2DModel(**model_config)
        model.load_state_dict(torch.load("working/fno-full-ddpm-ema-flowers-64/unet/diffusion_pytorch_model.bin"))

        pipeline = DDPMPipeline(
                      unet=model,
                      scheduler=noise_scheduler,
                   )
    else:
        pipeline = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32")
    pipeline.to("cuda:{}".format(rank))
    
    seed = torch.randint(high=int(1e9), size=(1,)).item()
    print("Seed: {}".format(seed))
    generator = torch.Generator(device=pipeline.device).manual_seed(torch.randint(high=int(1e9), size=(1,)).item())
    # run pipeline in inference (sample random noise and denoise)
    data = None
    batch_size = 16
    for i in range((num_samples + batch_size - 1) // batch_size):
        print("Batch {} of {}".format(i, (num_samples + batch_size - 1) // batch_size))
        images = pipeline(
            generator=generator,
            batch_size=batch_size,
            output_type="numpy",
        ).images
        data = images if data is None else np.concatenate((data, images))
    data = torch.tensor(data).transpose(2, 3).transpose(1, 2).numpy()
    if world_size == 1:
        dataset = Dataset.from_dict({"images": data})
        dataset.push_to_hub("Dahoas/pretrained-unet-cifar10-32")
    else:
        torch.save(dataset, "data_{}.pt".format(rank))


def sample_parallel():
    import torch.multiprocessing as mp

    world_size = 2
    num_samples = 32
    num_gpu_samples = num_samples // world_size
    result_queue = mp.Queue()
    for rank in range(world_size):
        mp.Process(target=sample_pipeline, args=(rank, world_size, False, num_gpu_samples, result_queue)).start()

    dataset = None
    for _ in range(world_size):
        sub_dataset = result_queue.get()
        dataset = sub_dataset if dataset is None else np.concatenate((dataset, sub_dataset))

    print(dataset.shape)
    dataset = Dataset.from_dict({"images": dataset})
    dataset.push_to_hub("Dahoas/pretrained-unet-cifar10-32")
    exit()


    #datasets = mp.spawn(sample_pipeline, args=(world_size, True, num_gpu_samples), nprocs=world_size, join=True)
    

def upload_pipeline():
    config_path = "configs/model_configs/original_config.yaml"
    model_path = "working/unet-ddpm-ema-flowers-64/unet/diffusion_pytorch_model.bin"
    pipeline = load_local_pipeline(config_path, model_path)
    unet = pipeline.unet
    unet.push_to_hub("Dahoas/unet-score")


if __name__ == "__main__":
    #upload_pipeline()
    #sample_pipeline(load_model=False)
    sample_parallel()