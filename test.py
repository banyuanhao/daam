import os
import shutil
import json
from mmcv.transforms import LoadImageFromFile

path = '/home/banyh2000/diffusion/daam/daam/dataset/ODFN/version_2/val/annotations/val_for_5_category_5_class_1_prompt.json'
# open json file
with open(path, 'r') as f:
    data = json.load(f)
    images = data['images']
    annotations = data['annotations']
    categories = data['categories']
print(categories)

# src_dir = "/home/banyh2000/diffusion/daam/dataset/ODFN/version_2/train/home/baseball_glove"
# dst_dir = "/home/banyh2000/diffusion/daam/dataset/ODFN/version_2/train/images/baseball_glove"

# # 确保目标目录存在
# os.makedirs(dst_dir, exist_ok=True)

# # 遍历源目录中的所有文件和文件夹
# for item in os.listdir(src_dir):
#     # 创建源和目标的完整路径
#     src_item = os.path.join(src_dir, item)
#     dst_item = os.path.join(dst_dir, item)

#     # 如果是文件夹，就移动它
#     if os.path.isdir(src_item):
#         shutil.move(src_item, dst_item)