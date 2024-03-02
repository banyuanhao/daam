from gpt4v_utils import gpt4_vision, gpt3_5_turbo, gpt4
from PIL import Image
from daam import trace, set_seed
import numpy as np
from diffusers import StableDiffusionPipeline
import torch
import matplotlib.pyplot as plt
import json
seed = 0
verbose = True
save = True
image_save = False
prompt_name = 'add2'
time = [5,15]
model_id = 'stabilityai/stable-diffusion-2-base'
device = 'cuda'
pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True).to(device)

base_path = '/home/banyh2000/diffusion/daam/wrapupdata/remove/remove_data'
with open(f'{base_path}/prompts_{prompt_name}.txt', 'r') as f:
    captions = f.readlines()
    captions = [caption.strip() for caption in captions]
    print(captions)
    
    
text_generate = f'# Task Assuming you are an object detector, please tell me the prominent foreground objects appears in the given image. The objects should be a part out of the coco detection dataset # Output Format Please directly respond with the names of the objects without adding any additional content or period. If there are multiple objects, please separate them with a comma. If there are no objects at all, please respond with none. # Example Input: A image Output: cat, potted plant Input: A image Output: none'

dict_failed = []
dict_success = []

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

def get_check(image_remove, negative_prompt):
    # text_check = f'tell me \'yes\' if the second image has removed at least one of the {negative_prompt} from the first image, otherwise, tell me \'no\'.'
    # context = gpt4_vision([image_ori, image_remove], text_check)
    # context_time = gpt4_vision([image_ori, image_remove_time], text_check)
    # context = 1 if 'yes' in context.lower() else 0
    # context_time = 1 if 'yes' in context_time.lower() else 0
    
    text_check = f'tell me \'yes\' if the image contains the {negative_prompt}, otherwise, tell me \'no\'.'
    context = gpt4_vision([image_remove], text_check)
    context = 1 if 'yes' in context.lower() else 0
    context = 1 - context
    
    return context

def get_related_words(prompt, word):
    text = f'Which word in the context of the "{prompt}" is most related to the word "{word}"?. Output the exact word without any additional content, without comma or period.'
    context = gpt4(text)
    return context
    
def experoment_one(id, seed):
    caption = get_caption(id)
    if verbose:
        print(caption)
    with torch.cuda.amp.autocast(dtype=torch.float16), torch.no_grad():
        output = pipe(caption, num_inference_steps=30, generator=set_seed(seed))
    image = output.images[0]
    
    objects = get_objects(image, caption)
    objects = filter_objects(caption, objects)
    if verbose:
        print(objects)
    
    for object in objects:
        time_id = [[0,3],[4,7],[8,11],[12,15],[16,19],[20,23],[24,27]]
        fig, axs = plt.subplots(2, len(time_id)+1, figsize=((len(time_id)+1)* 5,5*2))
        for row in axs:
            for ax in row:
                ax.axis('off')
        fig.tight_layout()
        anchor = get_related_words(caption, object)
        if anchor not in caption:
            continue
        with torch.cuda.amp.autocast(dtype=torch.float16), torch.no_grad():
            with trace(pipe) as tc:
                out = pipe(caption, num_inference_steps=30, generator=set_seed(seed))
                axs[0][0].imshow(out.images[0])
                for i, time_i in enumerate(time_id):
                    heat_map = tc.compute_global_heat_map(time_idx=time_i)
                    heat_map_word = heat_map.compute_word_heat_map(anchor)
                    heat_map_word.plot_overlay(out.images[0], ax=axs[0][i+1])
        with torch.cuda.amp.autocast(dtype=torch.float16), torch.no_grad():
            with trace(pipe) as tc:
                out = pipe(caption, negative_prompt=object, num_inference_steps=30, generator=set_seed(seed))
                axs[1][0].imshow(out.images[0])
                for i, time_i in enumerate(time_id):
                    heat_map = tc.compute_global_heat_map(time_idx=time_i)
                    heat_map_word = heat_map.compute_word_heat_map('n:'+object)
                    # convert 2D tensor heat_map_word.heatmap to PIL.Image
                    heat_map_word.plot_overlay(out.images[0], ax=axs[1][i+1])
        context = get_check(image, object)
        
        if context == 0:
            fig.savefig(f'{base_path}/images/failed/remove_{id}_{object}_{anchor}_{seed}.png')
            dict_failed.append({'id':id, 'caption':caption,'seed':seed, 'object':object, 'context':context})
            with open(f'{base_path}/images/failed.json', 'w') as f:
                json.dump(dict_failed, f)
        else:
            fig.savefig(f'{base_path}/images/success/remove_{id}_{object}_{anchor}_{seed}.png')
            dict_success.append({'id':id, 'caption':caption,'seed':seed, 'object':object, 'context':context})
            with open(f'{base_path}/images/success.json', 'w') as f:
                json.dump(dict_success, f)
        if verbose:
            print(context)
            
        # clear matplotlib plots
        plt.close(fig)

np.random.seed(seed)
for i in range(80):
    for j in range(3):
        seed_for_image = np.random.randint(0, 1000000)
        experoment_one(i, seed_for_image)
        