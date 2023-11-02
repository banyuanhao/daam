from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
model_id = "stabilityai/stable-diffusion-2"

device = "cuda" if torch.cuda.is_available() else "cpu"
# Use the Euler scheduler here instead
scheduler = EulerDiscreteScheduler.from_pretrained(
    model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(
    model_id, scheduler=scheduler, torch_dtype=torch.float16)
pipe = pipe.to(device)

prompt = "a photo of an astronaut riding a horse on mars"
negative_prompt = "blurry, dark photo, blue"
steps = 25
scale = 9
num_images_per_prompt = 1
seed = torch.randint(0, 1000000, (1,)).item()
generator = torch.Generator(device=device).manual_seed(seed)
image = pipe(prompt, negative_prompt=negative_prompt, width=768, height=768, num_inference_steps=steps,
             guidance_scale=scale, num_images_per_prompt=num_images_per_prompt, generator=generator).images[0]

image.save("astronaut_rides_horse.png")