# generating dataset with noise to labels mapping, only for 1 category
import json
from tqdm import tqdm
from utils_odfn import coco_classes, coco_classes_dict, seeds_spilt, seeds_plus_spilt, seeds_dict, seeds_plus_dict,extract_ground_seeds,extract_ground_seeds_plus,version_2_classes,version_1_classes

version = 'version_2'
if version == 'version_1':
    extract_ground = extract_ground_seeds
elif version == 'version_2':
    extract_ground = extract_ground_seeds_plus
else:
    raise ValueError('version should be version_1 or version_2')

if version == 'version_2':
    seeds_spilt = seeds_plus_spilt
    seeds_dict = seeds_plus_dict
    version_classes = version_2_classes
else:
    version_classes = version_1_classes
    
target_class = ['sports_ball']
#target_class = version_2_classes
target_id = [coco_classes_dict[i] for i in target_class]

for spilt in ['train', 'val', 'test']:
    seeds_sub = seeds_spilt(spilt)
        
    with open(f'dataset/ODFN/{version}/{spilt}/annotations/{spilt}.json', 'r') as f:
        data = json.load(f)
        images = data['images']
        annotations = data['annotations']
        categories = data['categories']
    annotations_for_1_category = []
    for annotation in annotations:
        image_id = annotation['image_id']
        category_id_truth, seed_id, prompt_id = extract_ground(image_id)
        score = annotation['score']
        category_id = annotation['category_id']
        bbox = annotation['bbox']
        id = annotation['id']
        rank = annotation['rank']
        if  category_id == category_id_truth and score > 0.6 and rank == 1 and category_id in target_id:
            dict_tmp= {}
            dict_tmp['image_id'] = seed_id
            dict_tmp['category_id'] = 0
            dict_tmp['bbox'] = [i/8 for i in bbox]
            dict_tmp['score'] = score
            dict_tmp['rank'] = rank
            dict_tmp['id'] = id
            dict_tmp['iscrowd'] = 0
            dict_tmp['area'] = annotation['area'] / 64
            annotations_for_1_category.append(dict_tmp)
    data['annotations'] = annotations_for_1_category
    print(len(annotations_for_1_category))

    images_for_1_category = []
    for seed in tqdm(seeds_sub):
        image_tmp = {}
        image_tmp['id'] = seeds_dict[seed]
        image_tmp['file_name'] = f'{spilt}/noises/' + str(seed) + '.png'
        image_tmp['width'] = 512 / 8
        image_tmp['height'] = 512 / 8
        images_for_1_category.append(image_tmp)
    data['images'] = images_for_1_category
    
    data['categories'] = [{'id': 0, 'name': 'object', 'supercategory': 'object'}]

    # save annotations
    with open(f'dataset/ODFN/{version}/{spilt}/annotations/{spilt}_for_1_category_1_class_png.json', 'w') as f:
        json.dump(data, f)
