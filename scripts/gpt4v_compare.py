from openai import OpenAI
import base64
import json
from tqdm import tqdm
from gpt4v_utils import gpt4_vision
import os
import argparse

parser = argparse.ArgumentParser(description='Diffusion')
parser.add_argument('--negative_prompt', type=str, default=None)
parser.add_argument('--negative_time', type=int, nargs = '+', default=None)
parser.add_argument('--seed_num', type=int, default=1000)
args = parser.parse_args()
negative_prompt = args.negative_prompt
negative_time = args.negative_time
seed_num = args.seed_num

target = f'negative_{negative_time[0]}_{negative_time[1]}'

base_path = "pics/removing/"
negative_path = base_path + negative_prompt.replace(" ","_")
class_names = ['no_negative', 'negative_0_30', target]
class_paths = [negative_path + '/' + class_name for class_name in class_names]

# read glasses.json
no_negative_id = None
if os.path.exists(f'{negative_path}/file.json'):
  print('loading no_negative_id')
  with open(f'{negative_path}/file.json', 'r') as f:
    data = json.load(f)
    no_negative_id = data['no_negative_id']

compare_answers = []
compare_answers_results = []

if os.path.exists(f'{negative_path}/{target}.json'):
  with open(f'{negative_path}/{target}.json', 'r') as f:
    data = json.load(f)
    compare_answers = data['compare_answers']
    length = len(compare_answers)//2
    compare_answers_results = data['compare_answers_results']
    compare_answers_results = compare_answers_results[:length]
    
    
dict_save = {}
dict_save['compare_answers'] = compare_answers
dict_save['compare_answers_results'] = compare_answers_results

text_prompt = 'which pic looks more similar to the first one, please answer with \'the second one\' or \'the third one\'.'


for i in tqdm(range(seed_num)):
  if str(i) in compare_answers:
    continue
  compare_answers.append(str(i))
  if no_negative_id != None and no_negative_id[i] == 0:
    compare_answers.append('NA')
    compare_answers_results.append(0)
  else:
    image_paths = [f'{class_path}/{i}.png' for class_path in class_paths]
    #try :
    context = gpt4_vision(image_paths, text_prompt)
    compare_answers.append(context)
    if 'second' in context:
      compare_answers_results.append(2)
    elif 'third' in context:
      compare_answers_results.append(3)
    else:
      raise ValueError('bad image')  
    # except:
    #   compare_answers.append('bad image')
      
    
# save compare_answers to file
  with open(f'{negative_path}/{target}.json', 'w') as f:
      json.dump(dict_save, f)
      
      
      
# text_prompt = 'Please answer three questions: Question one, considering the appearance, posture, and attire of the characters in the second and third pictures, which one resembles the first picture the most? Question two, considering the objects in the background of the second and third pictures, as well as their colors and arrangements, which one resembles the first picture the most? Question three, taking into account both of the above factors, which picture resembles the first picture the most? Please answer with "\'first\' or \'second\'," separated by a comma..'