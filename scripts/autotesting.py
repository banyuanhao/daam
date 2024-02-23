from gpt4v_utils import gpt4_vision, gpt3_5_turbo, gpt4
from generate_tools import generate_image
from PIL import Image
import json
import numpy as np
import os
seed = 0
verbose = True
save = True
image_save = False
prompt_name = 'add13'
time = [5,15]
dict_class = {}
dict = {}
if save:
    if os.path.exists(f'result_{seed}_{prompt_name}_{time[0]}_{time[1]}.json'):
        with open(f'result_{seed}_{prompt_name}_{time[0]}_{time[1]}.json', 'r') as f:
            dict = json.load(f)
    if os.path.exists(f'result_{seed}_class_{prompt_name}_{time[0]}_{time[1]}.json'):
        with open(f'result_{seed}_class_{prompt_name}_{time[0]}_{time[1]}.json', 'r') as f:
            dict_class = json.load(f)
        
with open('/home/banyh2000/diffusion/daam/daam/dataset/annotations/captions_train2017.json', 'r') as f:
    data = json.load(f)

with open(f'prompts_{prompt_name}.txt', 'r') as f:
    captions = f.readlines()
    captions = [caption.strip() for caption in captions]
    print(captions)
    
lenghth = len(data['annotations'])
    
text_generate = f'# Task Assuming you are an object detector, please tell me the prominent foreground objects appears in the given image. The objects should be a part out of the coco detection dataset # Output Format Please directly respond with the names of the objects without adding any additional content or period. If there are multiple objects, please separate them with a comma. If there are no objects at all, please respond with none. # Example Input: A image Output: cat, potted plant Input: A image Output: none'


text_compare = 'which pic looks more similar to the first one, please answer with \'the second one\' or \'the third one\'.'

def filter_objects(caption, objects):
    # remove repeated objects
    objects = list(set(objects))
    for object in objects:
        if object in caption.lower():
            objects.remove(object)
    if 'people' in objects:
        objects.remove('people')
    if 'person' in objects:
        objects.remove('person')
    return objects
    text = f"# Task remove the foreground objects in the object list that are described in the \'caption\' and return the rest of the objects. For example, if the caption describe a skier, then the object list shouldn't have the word people. Make sure the object is in coco dataset. # Output Format Please separate the left with a comma. If there are no objects left, please respond with \'none\'. # Example ## Input: object list: clock, potted plant, laptop, pencil, ground, wall Caption: Office wall above a wooden desk with potted plants on it. ## Output: clock, laptop # Input object list: {objects} caption: {caption}"
    context = gpt4(text)
    # remove the prefix of context, the prefix should be 'Output: ' or '# Output: '
    if context.startswith('Output: '):
        context = context[8:]
    elif context.startswith('# Output: '):
        context = context[10:]
    elif context.startswith('## Output: '):
        context = context[11:]
    else:
        pass
    return context.split(', ') if context != 'none' else []

def get_caption(id):
    '''Get the caption with the given id.'''
    # image_caption = data['annotations'][id]['caption']
    image_caption = captions[id]
    return image_caption

def get_original_image(caption, seed):
    image_ori = generate_image(prompt=caption, seed=seed, negative_prompt=None)
    return image_ori

def get_objects(image, caption):
    '''Get the objects in the image.'''
    context = gpt4_vision([image], text_generate)
    if 'none' in context:
        return []
    context = context.split(', ')
    # for con in context:
    #     if con in caption:
    #         context.remove(con) 
    return context

def get_removing_images(caption, negative_prompt, seed, negative_start=5, negative_end=15):
    image_remove_time = generate_image(prompt=caption, negative_prompt=negative_prompt, seed=seed, negative_time=list(range(negative_start,negative_end+1)))
    image_remove = generate_image(prompt=caption, negative_prompt=negative_prompt, seed=seed)
    return image_remove, image_remove_time

def get_check(image_ori, image_remove, image_remove_time, negative_prompt):
    # text_check = f'tell me \'yes\' if the second image has removed at least one of the {negative_prompt} from the first image, otherwise, tell me \'no\'.'
    # context = gpt4_vision([image_ori, image_remove], text_check)
    # context_time = gpt4_vision([image_ori, image_remove_time], text_check)
    # context = 1 if 'yes' in context.lower() else 0
    # context_time = 1 if 'yes' in context_time.lower() else 0
    
    text_check = f'tell me \'yes\' if the image contains the {negative_prompt}, otherwise, tell me \'no\'.'
    context = gpt4_vision([image_remove], text_check)
    context_time = gpt4_vision([image_remove_time], text_check)
    context = 1 if 'yes' in context.lower() else 0
    context_time = 1 if 'yes' in context_time.lower() else 0
    context = 1 - context
    context_time = 1 - context_time
    
    return context, context_time

def get_compare(image_ori, image_negative, image_negative_time):
    context = gpt4_vision([image_ori, image_negative, image_negative_time], text_compare)
    return 0 if 'second' in context else 1

def experoment_one(id, seed):
    caption = get_caption(id)
    if verbose:
        print(caption)
    image = get_original_image(caption, seed)
    if image_save:
        image.save('pics/image.png')
    objects = get_objects(image, caption)
    if verbose:
        print(objects)
    if len(objects) == 0:
        return 0, 0, 0, 0
    objects = filter_objects(caption, objects)
    if verbose:
        print(objects)
    if len(objects) == 0:
        return 0, 0, 0, 0
    
    
    
    context_total = 0
    context_time_total = 0
    compare_total = 0
    for object in objects:
        image_remove, image_remove_time = get_removing_images(caption, object, seed, time[0], time[1])
        if image_save:
            image_remove.save(f'pics/image_remove.png')
            image_remove_time.save(f'pics/image_remove_time.png')
        context, context_time = get_check(image, image_remove, image_remove_time, object)
        if object+'_negative' in dict_class.keys():
            dict_class[object+'_negative'] += context
            dict_class[object+'_negative_time'] += context_time
            dict_class[object+'_total'] +=1
        else:
            dict_class[object+'_negative'] = context
            dict_class[object+'_negative_time'] = context_time
            dict_class[object+'_total'] = 1
        if verbose:
            print(context, context_time)
        compare = get_compare(image, image_remove, image_remove_time)
        if verbose:
            print(compare)
        context_total += context
        context_time_total += context_time
        compare_total += compare
            
    return context_total, context_time_total, compare_total, len(objects)

np.random.seed(seed)
for i in range(80):
    if str(i) in dict.keys():
        continue
    id = i
    context_total, context_time_total, compare_total, total = 0, 0, 0, 0
    for j in range(5):
        seed_for_image = np.random.randint(0, 1000000)
        
        context, context_time, compare, total_ = experoment_one(id, seed_for_image)
        context_total += context
        context_time_total += context_time
        compare_total += compare
        total += total_
    dict[id] = {'context_total': context_total, 'context_time_total': context_time_total, 'compare_total': compare_total, 'total': total, 'seed': seed}
        
    # save dict as json
    if save:
        with open(f'result_{seed}_{prompt_name}_{time[0]}_{time[1]}.json', 'w') as f:
            json.dump(dict, f)
        with open(f'result_{seed}_class_{prompt_name}_{time[0]}_{time[1]}.json', 'w') as f:
            json.dump(dict_class, f)
        
    



