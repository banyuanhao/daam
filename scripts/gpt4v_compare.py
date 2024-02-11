from openai import OpenAI
import base64
import json
from tqdm import tqdm

# read glasses.json
with open('glasses.json', 'r') as f:
    data = json.load(f)
answers_id_no_negative = data['no_negative']

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

text_prompt = 'which pic looks more similar to the first one, please answer with the second or the third one.'


compare_answers = []
for i in tqdm(range(1000)):
  if answers_id_no_negative[i] == 0:
    compare_answers.append('NA')
  else:
    image_paths = [
      f'pics/removing/glasses/no_negative/{i}.png',
      f"pics/removing/glasses/negative/{i}.png",
      f'pics/removing/glasses/negative_6_12/{i}.png'
    ]
    context = gpt4_vision_compare(image_paths, text_prompt)
    compare_answers.append(context)
    
# save compare_answers to file
with open('compare_answers_6_12.json', 'w') as f:
    json.dump(compare_answers, f)
    
print(compare_answers)