# import os
# import shutil
# from mmcv.transforms import LoadImageFromFile
# import json

# path = '/home/banyh2000/diffusion/daam/daam/dataset/ODFN/version_2/val/annotations/val_for_5_category_5_class_1_prompt.json'
# # open json file
# with open(path, 'r') as f:
#     data = json.load(f)
#     images = data['images']
#     annotations = data['annotations']
#     categories = data['categories']
# print(categories)

# # src_dir = "/home/banyh2000/diffusion/daam/dataset/ODFN/version_2/train/home/baseball_glove"
# # dst_dir = "/home/banyh2000/diffusion/daam/dataset/ODFN/version_2/train/images/baseball_glove"

# # # 确保目标目录存在
# # os.makedirs(dst_dir, exist_ok=True)

# # # 遍历源目录中的所有文件和文件夹
# # for item in os.listdir(src_dir):
# #     # 创建源和目标的完整路径
# #     src_item = os.path.join(src_dir, item)
# #     dst_item = os.path.join(dst_dir, item)

# #     # 如果是文件夹，就移动它
# #     if os.path.isdir(src_item):
# #         shutil.move(src_item, dst_item)

# import torch
# from diffusers import StableDiffusionXLPipeline
# import matplotlib.pyplot as plt
# pipe = StableDiffusionXLPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
# )
# pipe = pipe.to("cuda")

# prompt = "a photo of an astronaut riding a horse on mars"
# image = pipe(prompt).images[0]
# plt.clf()
# plt.rcParams.update({'font.size': 12})

# fig, axs = plt.subplots(4,4)
# plt.subplots_adjust(top=0.95) 
 
# axs[0][0].imshow(image)
# plt.savefig("pics/pic.png")

# from diffusers import AutoPipelineForText2Image
# import torch
# import matplotlib.pyplot as plt

# pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
#     "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
# ).to("cuda")

# prompt = "Astronaut in a jungle"
# image = pipeline_text2image(prompt=prompt).images[0]
# import numpy as np
# from PIL import Image
# # 将 Image 对象转换为像素数组
# image_array = np.array(image)

# # 将像素值缩放到 0 到 1 的范围
# image_array = image_array / 256.0

# # 将像素数组转换回 Image 对象
# image = Image.fromarray(image_array)
# plt.imsave("pics/pic.png",image)


import matplotlib.pyplot as plt
import numpy as np
#  0 1  2  3  4 5 6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
# 11 9 13 11 10 9 9 10 11 13 12 17 -1 -1 -1 -1 -1 -1 -1 -1 -1

#  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
#  9 10 10 11 10 11 11 12 13 16 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
stop = [11, 9, 13, 11, 10, 9, 9, 10, 11, 13, 12, 17]
stop = [9, 10, 10, 11, 10, 11, 11, 12, 13, 16]
pos = np.arange(15)

plt.plot(pos[:len(stop)],stop - pos[:len(stop)])
plt.savefig("pics/pic.png")

