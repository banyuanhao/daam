from utils_odfn import seeds_plus, seeds_plus_spilt, seeds_plus_dict, seeds_plus_shuffled_dict
import random
import numpy as np
import json


# tmp = seeds_plus.copy()
# sub = tmp[:17500]
# random.shuffle(sub)
# print(len(sub))
# tmp[:17500] = sub
# sub = tmp[17500:18500]
# random.shuffle(sub)
# tmp[17500:18500] = sub
# sub = tmp[18500:]
# random.shuffle(sub)
# tmp[18500:] = sub
# np.save('scripts/odfn/generating/seeds_plus_shuffle.npy', tmp)

path_src = 'dataset/ODFN/version_2/{}/annotations/{}_for_1_category_1_class.json'
path_tar = 'dataset/ODFN/version_2/{}/annotations/{}_for_1_category_1_class_shuffled.json'

for spilt in ['train', 'val', 'test']:
    with open(path_src.format(spilt,spilt), 'r') as f:
        data = json.load(f)
        images = data['images']
        annotations = data['annotations']
        categories = data['categories']
        
    for image in images:
        seed = image['file_name'].split('/')[-1].split('.')[0]
        image['file_name'] = f'{spilt}/noises/' + str(seeds_plus_shuffled_dict[int(seed)]) + '.npy'
    data['images'] = images
        
    with open(path_tar.format(spilt,spilt), 'w') as f:
        json.dump(data, f)
