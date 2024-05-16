# import os

# path = 'wrapupdata/moreusage/FID/natural'
# names = os.listdir(path)
# for name in names:
#     if name.endswith('.JPG'):
#         os.rename(os.path.join(path, name), os.path.join(path, name[:-4] + '.jpg')
#                    )

# import os
# from PIL import Image

# path = 'wrapupdata/moreusage/FID/cat/raw'
# path_target =  'wrapupdata/moreusage/FID/cat/normalize'
# names = os.listdir(path)

# for i,name in enumerate(names):
#     image_path = os.path.join(path, name)
#     target_path = os.path.join(path_target, f'{i}.jpg')
#     ### read the image, resize it to 512x512, and save it to the target path
#     img = Image.open(image_path)
#     ### crop 
#     if img.size[0] > img.size[1]:
#         n = (img.size[0] - img.size[1]) // 2
#         img = img.crop((n, 0, n+ img.size[1], img.size[1]))
#     else:
#         n = (img.size[1] - img.size[0]) // 2
#         img = img.crop((0, n, img.size[0], img.size[0]+n))
#     img = img.resize((512, 512))
#     if img.mode != 'RGB':
#         img = img.convert('RGB')
#     img.save(target_path)
import os
path_source = 'wrapupdata/moreusage/FID/cat/generated/31'
path_target =  'wrapupdata/moreusage/FID/cat/generated/31_spilt'
os.makedirs(path_target, exist_ok=True)



import os
# os.makedirs(path_target, exist_ok=True)
names = os.listdir(path_source)
for name in names:
    adj = name.split('_')[1].split('.')[0]
    prompt = name.split('_')[0]
    os.rename(os.path.join(path_source, name), os.path.join(os.path.join(path_target,adj), name[:-4] + f'{prompt}.png'))
    