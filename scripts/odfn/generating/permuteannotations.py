from utils_odfn import seeds_plus, seeds_plus_spilt, seeds_plus_shuffled_dict,seeds_plus_dict
import random
import numpy as np
import json
# print(seeds_plus_shuffled_dict)
# tmp = []
# sub_seed = seeds_plus_spilt('train')
# random.shuffle(sub_seed)
# tmp.extend(sub_seed)
# sub_seed = seeds_plus_spilt('val')
# random.shuffle(sub_seed)
# tmp.extend(sub_seed)
# sub_seed = seeds_plus_spilt('test')
# random.shuffle(sub_seed)
# tmp.extend(sub_seed)
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
