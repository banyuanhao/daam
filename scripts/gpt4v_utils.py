import base64
import requests
import random
import time
import concurrent
import os
from scripts.convert_to_json import convert_format
from collections import defaultdict
import json


api_key = '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'


# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


def gpt4v(image_paths, prompt="What’s in this image?"):
  if not isinstance(image_paths, list): image_paths = [image_paths]
  ## prepare image for GPT4V to process
  api_key = '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'
  base64_images = [encode_image(image_path) for image_path in image_paths]

  content = [
    {
      "type": "text",
      "text": prompt
    }
  ]
  for base64_image in base64_images:
    content.append({
      "type": "image_url",
      "image_url": {
        "url": f"data:image/jpeg;base64,{base64_image}"
      }
    })

  headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
  }

  payload = {
    "model": "gpt-4-vision-preview",
    "messages": [
      {
        "role": "user",
        "content": content
      }
    ],
    "max_tokens": 50
  }

  while True:
    try:
      response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
      ret = response.json()['choices'][0]['message']['content']
      break
    except Exception as e:
      print(f'ERROR: {e}')
      time.sleep(5)
      continue
  print(ret)

  return ret


def gpt4v_parallel(images, prompt="What’s in this image?", max_workers=64):
  list_of_kwargs = [(image, prompt) for image in images]
  llm_fn = gpt4v

  ## probe
  ret = llm_fn(*list_of_kwargs[0])
  print('probe successfully')

  responses = run_in_parallel(
    function=llm_fn,
    inputs=list_of_kwargs,
    max_workers=min(max_workers, len(images)),
    report_progress=True
  )

  return responses


def run_in_parallel(function, inputs, max_workers=-1, report_progress=True):
  if max_workers == -1:
    max_workers = len(inputs)
  
  if len(inputs) == 1:  ## single process mode
    ret = function(*inputs[0])
    if not isinstance(ret, list):
      ret = [ret]
  else:
    ret = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
      results = list(executor.map(function, *zip(*inputs)))
      if isinstance(results[0], list):
        for result in results:
          ret += result
      else:
        ret = results

  return ret


def generate_contrastive_list(original_list):
  contrastive_list = []

  # Create a dictionary to group images by label
  label_dict = {}
  for img, label in original_list:
    if label not in label_dict:
      label_dict[label] = []
    label_dict[label].append(img)

  # Generate contrastive pairs
  for img, label in original_list:
    # get all contrastive labels
    labels_cont = [l for l in label_dict.keys() if l != label]
    
    # Randomly select an image from each label
    imgs_cont = []
    for label_cont in labels_cont:
      imgs_cont.append(random.choice(label_dict[label_cont]))

    # Append to contrastive list
    contrastive_list.append(([img, *imgs_cont], label))

  return contrastive_list


def generate_caption_dataset(dataset, prompt, mode='v1'):
  """
  Generates a caption dataset based on the given dataset and prompt.

  Args:
    dataset (list): A list of (image_path, label_name)
    prompt (str): The prompt to generate captions for the images.

  Returns:
    list: A list of tuples containing captions and corresponding labels.
  """

  assert isinstance(dataset[0][-1], str), 'dataset should be a list of (image_path, label_name).'
  caption_dataset = []

  if mode == 'v1':
    imgs = [img for img, _ in dataset]
    captions = gpt4v_parallel(imgs, prompt)
  elif mode == 'v2':  # contrastive
    dataset_cont = generate_contrastive_list(dataset)
    imgs = [img for img, _ in dataset_cont]
    captions = gpt4v_parallel(imgs, prompt)
  assert len(captions) == len(imgs)

  for caption, (_, label_name) in zip(captions, dataset):
    caption_dataset.append((caption, label_name))
  return caption_dataset


def build_fgvc(cap_subsets, split, bench='vision'):
  """
  Build FGVC dataset.

  """

  for label_group in cap_subsets:
    data_list = cap_subsets[label_group]['data_list']
    label_names = cap_subsets[label_group]['label_names']
  
    convert_format(data_list, bench, label_group, split, label_names=label_names)


def load_caption(caption_file):
  with open(caption_file, 'r') as f:
    data = json.load(f)
  return data


def save_caption(caption_file, caption_dataset):
  with open(caption_file, 'w') as f:
    json.dump(caption_dataset, f, indent=4)


def dataset_to_list(dataset):
  """preprocess the dataset into a list of (image_path, label_name) pairs
  """
  idx_to_label = {v: k for k, v in dataset.class_to_idx.items()}
  dataset_list = []
  for image_path, label_idx in dataset._samples:
    dataset_list.append((image_path, idx_to_label[label_idx]))
  return dataset_list


def retrieve_subset(data_list, label_names, num_data_per_label):
  subset_dict = defaultdict(list)
  for item in data_list:
    _, label = item
    if label in label_names:
      subset_dict[label].append(item)

  subset_list = []
  for label in subset_dict:
    if len(subset_dict[label]) < num_data_per_label:
      return None
    subset_list += subset_dict[label][:num_data_per_label]
  return subset_list


def make_caption_dataset(subsets, caption_file_template, prompt="What's in the image?", mode='v1'):
  caption_subsets = {}
  for label_group in subsets:
    caption_file = caption_file_template.format(label_group)
    # reuse_caption = os.path.exists(caption_file) and input(f'{caption_file} exists. Override? [y/n]') == 'n'
    reuse_caption = os.path.exists(caption_file) and False

    if reuse_caption:
      print('Reuse caption file.')
      caption_subsets[label_group] = load_caption(caption_file)
    else:
      print('Generate caption file.')
      caption_subsets[label_group] = {
        'data_list': generate_caption_dataset(subsets[label_group]['data_list'], prompt, mode=mode),  # captions
        'label_names': subsets[label_group]['label_names']
      }
      save_caption(caption_file, caption_subsets[label_group])

  return caption_subsets
