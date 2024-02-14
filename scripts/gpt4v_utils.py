from openai import OpenAI
import base64
import io
from PIL import Image

with open('/home/banyh2000/openaiorgkey.txt') as file:
    OPENAI_ORG = file.read()
# read api key from ~/openaiapi.txt
with open('/home/banyh2000/openaiapi.txt') as file:
    OPENAI_API_KEY = file.read()
client = OpenAI(api_key=OPENAI_API_KEY,organization=OPENAI_ORG)

    
# Function to encode the image
def encode_image(image_path):
  if type(image_path) == str:
    with open(image_path, "rb") as image_file:
      return base64.b64encode(image_file.read()).decode('utf-8')
  else: # PIL image
    buffered = io.BytesIO()
    image_path.save(buffered, format="JPEG")
    img_str = buffered.getvalue()
    return base64.b64encode(img_str).decode('utf-8')


def gpt4_vision(images, prompt_text):
  base64_images = [encode_image(image) for image in images]
  
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

def gpt3_5_turbo(prompt_text):
      
  response = client.chat.completions.create(
    model="gpt-3.5-turbo-0125",
    messages=[
      {
        "role": "user",
        "content": prompt_text
      }
    ],
    max_tokens=300,
  )
  return(response.choices[0].message.content)

def gpt4(prompt_text):
      
  response = client.chat.completions.create(
    model="gpt-4",
    messages=[
      {
        "role": "user",
        "content": prompt_text
      }
    ],
    max_tokens=300,
  )
  return(response.choices[0].message.content)