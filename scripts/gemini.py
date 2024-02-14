from scripts.gemini_detection import GeminiAPIWrapper, GeminiVisionAPIWrapper
from scripts.gemini_detection import genai
import PIL.Image as Image
from tqdm import tqdm
import numpy as np
with open('/home/banyh2000/googleapi.txt') as file:
    GOOGLE_API_KEY = file.read()

genai.configure(api_key=GOOGLE_API_KEY)
model = GeminiVisionAPIWrapper('gemini-pro-vision',max_tokens=50)

with open('glasses_no_negative.txt', 'r') as file:
    answers_no_negative = file.readlines()
answers_no_negative = [answer.strip() for answer in answers_no_negative]
answers_id_no_negative = [1 if 'yes' in answer else 0 for answer in answers_no_negative]

with open('glasses_negative_time.txt', 'r') as file:
    answers_negative_time = file.readlines()
answers_negative_time = [answer.strip() for answer in answers_negative_time]
answers_id_negative_time = [1 if 'yes' in answer else 0 for answer in answers_negative_time]

with open('glasses_negative.txt', 'r') as file:
    answers_negative = file.readlines()
answers_negative = [answer.strip() for answer in answers_negative]
answers_id_negative = [1 if 'yes' in answer else 0 for answer in answers_negative]

with open('glasses_baseline.txt', 'r') as file:
    answers_baseline = file.readlines()
answers_baseline = [answer.strip() for answer in answers_baseline]
answers_id_baseline = [1 if 'yes' in answer else 0 for answer in answers_baseline]

import json
dict_removing = {
    "no_negative": answers_id_no_negative,
    "negative": answers_id_negative,
    "negative_time": answers_id_negative_time,
    "baseline": answers_id_baseline
}
with open('file.json', 'w') as f:
    json.dump(dict_removing, f)
    
with open('file.json', 'r') as f:
    data = json.load(f)
answers_id_baseline = data['baseline']
answers_id_negative = data['negative']
answers_id_negative_time = data['negative_time']
answers_id_no_negative = data['no_negative']

total = np.sum(answers_id_no_negative)
total_negative = np.sum(answers_id_negative)
total_negative_time = np.sum(answers_id_negative_time)
total_baseline = np.sum(answers_id_baseline)
print(total)
print(total_negative)
print(total_negative_time)
print(total_baseline)