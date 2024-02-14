from openai import OpenAI
import base64
import json
from tqdm import tqdm
import os
import argparse

parser = argparse.ArgumentParser(description='Diffusion')
parser.add_argument('--negative_prompt', type=str, default=None)
parser.add_argument('--negative_time', type=int, nargs = '+', default=None)
args = parser.parse_args()
negative_prompt = args.negative_prompt
negative_time = args.negative_time

# read api key from ~/openaiapi.txt
with open('/home/banyh2000/openaiapi.txt') as file:
    OPENAI_API_KEY = file.read()
client = OpenAI(api_key=OPENAI_API_KEY)
    
# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def gpt4_vision_compare(image_paths, prompt_text):
  base64_images = [encode_image(image_path) for image_path in image_paths]
  content = [
  {
      "type": "text",
      "text": prompt_text
  }
  ]

  for base64_image in base64_images:
      content.append({
          "type": "image_url",
          "image_url": {
          "url": f"data:image/jpeg;base64,{base64_image}"
          }
      })
      
  response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
      {
        "role": "user",
        "content": content
      }
    ],
    max_tokens=300,
  )
  return(response.choices[0].message.content)

text_prompt = f'tell me \'yes\' if there is {negative_prompt} in the picture, otherwise, tell me \'no\'.'

target = f'negative_{negative_time[0]}_{negative_time[1]}' if negative_time is not None else 'no_negative'

base_path = "pics/removing/"
negative_path = base_path + negative_prompt.replace(" ","_")
time_path = negative_path + '/' + target

answers_id_no_negative = None
answers_target = []
answers_id_target = []

data = {}
if os.path.exists(f'{negative_path}/file.json'):
  with open(f'{negative_path}/file.json', 'r') as f:
      data = json.load(f)
  if 'no_negative' in data.keys():
    answers_id_no_negative = data['no_negative_id']
    answers_no_negative = data['no_negative']
  if target in data.keys():
    answers_target = data[f'{target}']
    length = len(answers_target) // 2
    answers_id_target = data[f'{target}_id'][:length]
  
    
for i in tqdm(range(1000)):
  if str(i) in answers_target:
    continue
  answers_target.append(str(i))
  if target != 'no_negative' and answers_id_no_negative is not None and answers_id_no_negative[i] == 0:
    answers_target.append('NA')
    answers_id_target.append(0)
  else:
    image_paths = [
      f'{time_path}/{i}.png',
    ]
    # try :
    context = gpt4_vision_compare(image_paths, text_prompt)
    answers_target.append(context.lower())
    if 'yes' in context.lower():
      answers_id_target.append(1)
    else:
      answers_id_target.append(0)
    # except:
    #   print(context)
    #   answers_target.append('bad image')
    #   answers_id_target.append(0)
      
      
  data[f'{target}_id'] = answers_id_target
  data[f'{target}'] = answers_target

  # save compare_answers to file
  with open(f'{negative_path}/file.json', 'w') as f:
      json.dump(data, f)