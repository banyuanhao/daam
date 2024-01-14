import cv2
from mmcv.transforms.loading import LoadImageFromNPY
from mmdet.datasets.transforms.loading import LoadMultiChannelImageFromFiles
import numpy as np
import mmcv


# get a random np array
img = np.random.randint(0,255,size = (64, 64))
# save img as a np array
np.save('pics/test.npy', img)
# save img as png using mmcv
# mmcv.imwrite(img, 'pics/test1.png')

# instance_ = {'img_prefix': '/content/drive/MyDrive/test_data',
#             'img_path': ['/home/banyh2000/diffusion/daam/daam/dataset/ODFN/train/images/apple/29403/apple_29403_3.png', '/home/banyh2000/diffusion/daam/daam/dataset/ODFN/train/images/apple/29403/apple_29403_2.png']}

# instance_ = {'img_prefix': '/content/drive/MyDrive/     test_data','img_path': '/home/banyh2000/diffusion/daam/daam/pics/test.npy'}
# transform = LoadImageFromNPY()
# instance_ = transform(instance_)
# print(type(instance_['img']))

instance_ = {'img_prefix': '/content/drive/MyDrive/     test_data','img_path': ['/home/banyh2000/diffusion/daam/daam/pics/test1.png','/home/banyh2000/diffusion/daam/daam/pics/test1.png']}
transform = LoadMultiChannelImageFromFiles()
instance_ = transform(instance_)
print(type(instance_['img']))