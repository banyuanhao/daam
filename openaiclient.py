# key 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
from openai import OpenAI
client = OpenAI()

completion = client.chat.completions.create(
  model="gpt-4",
  messages=[
    {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
    {"role": "user", "content": "Can you generate a description using the word: cow? the scentence should be no more than 15 words. And you may adding some other objects to the scene. For example: A dairy cow is grazing beside a fence, with a tree in the distance.  Please generate 10 sentences for the word."}
  ]
)

completion = client.chat.completions.create(
  model="gpt-4",
  messages=[
    {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
    {"role": "user", "content": "Can you generate a description using the word: cow? the scentence should be no more than 15 words. And you may adding some other objects to the scene. For example: A dairy cow is grazing beside a fence, with a tree in the distance.  Please generate 10 sentences for the word."},
    completion.choices[0].text
  ]
)
print(completion.choices[0].message)