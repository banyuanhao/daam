from daam import trace, set_seed
from diffusers import StableDiffusionPipeline
from matplotlib import pyplot as plt
import torch
import random
import os
os.environ["VISIBLE_CUDA_DEVICES"] = "6"

gen = set_seed(0)  # for reproducibility

numbers = [random.randint(1, 10000000) for _ in range(10)]
print(numbers)

a = torch.tensor([10,10,20])
print(a)
b = a.mean(0)
b[1] = 0
print(b)
print(a)