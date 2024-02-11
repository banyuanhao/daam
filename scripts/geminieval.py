import google.generativeai as genai
import PIL.Image as Image
from tqdm import tqdm

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

if __name__ == "__main__":
    model = GeminiVisionAPIWrapper('gemini-pro-vision',max_tokens=50)

    # experiment, tell if there is glasses
    #answers = []
    prompt_text = "tell me yes if the woman is wearing a pair of glasses"
    base_path =  '/home/banyh2000/diffusion/daam/pics/removing/glasses/'    

    # for i in tqdm(range(1000)):
    #     img_path = base_path + f'no_negative/{i}.png'
    #     img = Image.open(img_path)
    #     response = model([prompt_text,img])
    #     answers.append(response.lower())
        

    # # write answers to file
    # with open('glasses_no_negative.txt', 'w') as file:
    #     for answer in answers:
    #         file.write(answer + '\n')

    # read answers from file
    # with open('glasses_no_negative.txt', 'r') as file:
    #     answers_negative = file.readlines()
    # answers_no_negative = [answer.strip() for answer in answers_no_negative]

    # answers_negative = []
    # for i in tqdm(range(1000)):
    #     if answers_id[i] == 0:
    #         answers_negative.append('NA')
    #     else:
    #         img_path = base_path + f'negative/{i}.png'
    #         img = Image.open(img_path)
    #         response = model([prompt_text,img])
    #         answers_negative.append(response.lower())
            
    # # write answers to file
    # with open('glasses_negative.txt', 'w') as file:
    #     for answer in answers_negative:
    #         file.write(answer + '\n')

    # read answers from file
    # with open('glasses_negative.txt', 'r') as file:
    #     answers_negative = file.readlines()
    # answers_negative = [answer.strip() for answer in answers_negative]

    # answers_negative_time = []
    # for i in tqdm(range(1000)):
    #     if answers_id[i] == 0:
    #         answers_negative_time.append('NA')
    #     else:
    #         img_path = base_path + f'negative_time/{i}.png'
    #         img = Image.open(img_path)
    #         response = model([prompt_text,img])
    #         answers_negative_time.append(response.lower())
            
    # write answers to file
    # with open('glasses_negative_time.txt', 'w') as file:
    #     for answer in answers_negative_time:
    #         file.write(answer + '\n')

    # read answers from file
    # with open('glasses_negative.txt', 'r') as file:
    #     answers_negative_time = file.readlines()
    # answers_negative_time = [answer.strip() for answer in answers_negative_time]



    # img1 = Image.open('/home/banyh2000/diffusion/daam/pics/removing/glasses/no_negative/900.png')
    # img2 = Image.open('/home/banyh2000/diffusion/daam/pics/removing/glasses/negative/900.png')
    # img3 = Image.open('/home/banyh2000/diffusion/daam/pics/removing/glasses/negative_time/900.png')
    # model = GeminiVisionAPIWrapper('gemini-pro-vision',max_tokens=50)
    # prompt = [
    #         "tell me yes if the woman wearing glasses", 
    #         img1,img2
    #     ]
    # prompt = [
    #         "tell me which pic the second or the third one looks more similar to the first one.", 
    #         img1,img2,img3
    #     ]
    # response = model(prompt)
    # print(response)


    # answers_baseline = []
    # for i in tqdm(range(1000)):
    #     img_path = base_path + f'baseline/{i}.png'
    #     img = Image.open(img_path)
    #     response = model([prompt_text,img])
    #     answers_baseline.append(response.lower())


    # # write answers to file
    # with open('glasses_baseline.txt', 'w') as file:
    #     for answer in answers_baseline:
    #         file.write(answer + '\n')

            
    # # read answers from file
    # with open('glasses_baseline.txt', 'r') as file:
    #     answers_baseline = file.readlines()
    # answers_baseline = [answer.strip() for answer in answers_baseline]
    # answers_id_baseline = [1 if 'yes' in answer else 0 for answer in answers_baseline]

    # read answers from file
    with open('glasses_no_negative.txt', 'r') as file:
        answers_no_negative = file.readlines()
    answers_no_negative = [answer.strip() for answer in answers_no_negative]
    answers_id_no_negative = [1 if 'yes' in answer else 0 for answer in answers_no_negative]

    answers_negative_6_12 = []
    for i in tqdm(range(1000)):
        if answers_id_no_negative[i] == 0:
            answers_negative_6_12.append('NA')
        else:
            img_path = base_path + f'baseline/{i}.png'
            img = Image.open(img_path)
            response = model([prompt_text,img])
            answers_negative_6_12.append(response.lower())
    
    # write answers to file
    with open('glasses_negative_6_12.txt', 'w') as file:
        for answer in answers_negative_6_12:
            file.write(answer + '\n')
            