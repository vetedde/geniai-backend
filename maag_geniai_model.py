import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, DDIMScheduler

def generate_image(prompt):
    model_path = '/home/user/PycharmProjects/genai/4500' #path to model weights

    pipe = StableDiffusionPipeline.from_pretrained(model_path, safety_checker=None, torch_dtype=torch.float16).to(
        "cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()

    g_cuda = torch.Generator(device='cuda')
    seed = 52362
    g_cuda.manual_seed(seed)

    prompt = prompt
    negative_prompt = ""
    num_samples = 1
    guidance_scale = 7.5
    num_inference_steps = 150
    height = 512
    width = 512
    with autocast("cuda"), torch.inference_mode():
        images = pipe(
            prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_samples,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=g_cuda
        ).images

    images[0].save("img.jpg")

    # Save the image and return the file path
    image_path = "img.jpg"


    return image_path
