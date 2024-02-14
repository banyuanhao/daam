from gpt4v_utils import gpt4_vision

negative_prompt = 'potted_plant'
seed = 200

image_ori = f'pics/removing/{negative_prompt}/no_negative/{seed}.png'
image_remove = f'pics/removing/{negative_prompt}/negative_0_30/{seed}.png'
image_remove_time = f'pics/removing/{negative_prompt}/negative_6_12/{seed}.png'
negative_prompt = negative_prompt.replace('_',' ')
text_check = f'tell me \'yes\' if the second image has removed at least one of the {negative_prompt} from the first image, otherwise, tell me \'no\'.'
context = gpt4_vision([image_ori, image_remove], text_check)
context_time = gpt4_vision([image_ori, image_remove_time], text_check)
print(context)
print(context_time)
