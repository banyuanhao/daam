import google.generativeai as genai
import PIL.Image as Image
from tqdm import tqdm
import json
from tqdm import tqdm
import os
import argparse

with open('/home/banyh2000/googleapi.txt') as file:
    GOOGLE_API_KEY = file.read()

genai.configure(api_key=GOOGLE_API_KEY)

# Define a wrapper around the GPT-4 API to match the interface you need.
class GeminiAPIWrapper:
    def __init__(self, model_name="gemini", max_tokens=30):
        
        # Support for attack framework
        self.name = "google-gemini"

        # Configurable model params
        self.model_name = model_name + "-pro"
        self.max_tokens = max_tokens
    
    def __call__(self, prompt_list): # gpt-3.5-turbo

        model = genai.GenerativeModel(self.model_name)
        chat = model.start_chat(history=[])
        print("Calling Gemini ...")
        for idx, prompt in enumerate(prompt_list):
            try:
                response = chat.send_message(prompt)
            except Exception as e:
                pass
                print("Gemini refuse to anwer!!!")

                return "Sorry, gemini refuse to answer."
        response.resolve()
        return response.text
    
class GeminiVisionAPIWrapper:
    def __init__(self, model_name="gemini-pro-vision", max_tokens=30):
        
        # Support for attack framework
        self.name = "google-gemini"

        # Configurable model params
        self.model_name = model_name
        self.max_tokens = max_tokens
    
    def __call__(self, prompt_list): # gpt-3.5-turbo

        model = genai.GenerativeModel(self.model_name)
        response = model.generate_content(prompt_list, stream=True)
        response.resolve()
        return response.text
    
    def complicated(self, prompt_list):
        model = genai.GenerativeModel(self.model_name)
        response = model.generate_content(prompt_list, stream=True)
        response = response._result.parts
        return response

    

parser = argparse.ArgumentParser(description='Diffusion')
parser.add_argument('--negative_prompt', type=str, default=None)
parser.add_argument('--negative_time', type=int, nargs = '+', default=None)
args = parser.parse_args()
negative_prompt = args.negative_prompt
negative_time = args.negative_time

model = GeminiVisionAPIWrapper('gemini-pro-vision',max_tokens=50)

text_prompt = f'tell me \'yes\' if there is {negative_prompt} in the picture, otherwise, tell me \'no\'.'
target = f'negative_{negative_time[0]}_{negative_time[1]}' if negative_time is not None else 'no_negative'


base_path = "pics/removing/"
negative_path = base_path + negative_prompt
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
    image_path = f'{time_path}/{i}.png'
            
    img = Image.open(image_path)
    # try :
    context = model([text_prompt,img])
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
        