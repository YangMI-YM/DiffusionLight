# predownload SDXL model
import torch
from transformers import pipeline
from diffusers import AutoencoderKL, StableDiffusionXLControlNetInpaintPipeline, ControlNetModel
  
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
controlnet = ControlNetModel.from_pretrained(
      "diffusers/controlnet-depth-sdxl-1.0",
      torch_dtype=torch.float16,
      variant="fp16",
)

pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
      "stabilityai/stable-diffusion-xl-base-1.0",
      controlnet=controlnet,
      torch_dtype=torch.float16,
      variant="fp16"
)
depth_estimator = pipeline(task="depth-estimation", model="Intel/dpt-hybrid-midas")
# free up memory after preload sdxl
del vae
del controlnet
del pipe
del depth_estimator
