from daam import trace, set_seed
from models.diffuserpipeline import StableDiffusionPipelineForNegativePrompts
from matplotlib import pyplot as plt
import torch
import random
import os

repo_id = "runwayml/stable-diffusion-v1-5"


# replace `dpm` with any of `ddpm`, `ddim`, `pndm`, `lms`, `euler_anc`, `euler`
pipeline = StableDiffusionPipelineForNegativePrompts.from_pretrained()