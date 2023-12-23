import base64
import os
import requests
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv
import time
import uuid
from glob import glob
import json
import pandas as pd
import re
import argparse
from tqdm import tqdm

def get_sort_key(x: str)->int:
    page, image = x.split('/')[-1].strip('.png').strip('p').split('-')
    sort_key = int(page) * 1000000 + int(image) # 1000000 for just a big number
    return sort_key

def split_text_by_pattern(text: str)->list:
    '''
    Split text which has the pattern of '도 k' or '도k' where k is integer
    '''
    pattern = r'(도 ?\d+)'
    # Split the text at each occurrence of the pattern
    split_text = re.split(pattern, text)

    # The first element in the list might be an empty string, so remove it if it is
    split_text = [i for i in split_text if i]

    # Combine the split parts back into complete sentences
    result = [''.join(split_text[i:i+2]) for i in range(0, len(split_text), 2)]

    return result

def encode_image(image_path: str)->str:
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def ocr_call(base64_image:str, url:str, headerDict:dict)->tuple:
    datas = {
    'version': "V2",
    "images": [
      {
        "format": "png",
        "name": "medium",
        "data": base64_image
      }
    ],
    "lang": "ko",
    "requestId": str(uuid.uuid4()),
    'timestamp': int(round(time.time() * 1000))
    }
    response_ocr = requests.post(url, json = datas, headers = headerDict) # API POST
    response_processed = dict(response_ocr.json()) 
    ocr_result_korean = list()
    ocr_result_numbers = list()
    def contains_numbers(input_string):
        num_digit = 0
        num_non_digit = 0
        for char in input_string:
            if char.isdigit(): # If given string includes digits, retrun True
                num_digit += 1
            else: num_non_digit += 1
        if num_digit >= num_non_digit: return True
        else: return False
    for item in response_processed['images'][0]['fields']:
        item['inferText'] = item['inferText'].replace('-', '')
        if contains_numbers(item['inferText']): # if it contains numbers, append to ocr_result_numbers
            ocr_result_numbers.append(item['inferText'])
        else: # if it does not contain numbers, append to ocr_result_korean
            ocr_result_korean.append(item['inferText'])
    ocr_result_numbers = list(set(ocr_result_numbers)) # remove duplicates
    return ocr_result_korean, ocr_result_numbers

def gpt_call(ocr_result_korean:str, ocr_result_numbers:str, base64_image:str)->str:
    prompt = f"Describe the image which is a drawing from a patent only in Korean. Sometimes a drawing may include Korean. \
          If you are referring to the Korean word in the image, use these vocabularies ({ocr_result_korean})\
          In your description, include \
          1. explanation of what the image looks like in one or two sentences.\
          2. explanation of parts pointed at or labeled by certain numbers or strings.\
          3. a list of all numbers in the image\
          Below is an example output you can refer to:\
          ```\
          The illustration depicts <description>.\
          \
          Explanation of parts pointed at or labeled by certain numbers ({ocr_result_numbers}):\
          1. Number or letter <number or string> might indicate/ is pointing to/ appears to be associated with / is near <description>.\
          2. Number or letter <number or string> might indicate/ is pointing to/ appears to be associated with / is near <description>.\
          3. Number or letter <number or string> might indicate/ is pointing to/ appears to be associated with / is near <description>.\
          ...\
          Images contains [<list of numbers in the image>]\
                    ```"
    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
    }

    payload = {
    "model": "gpt-4-vision-preview",
    "messages": [
        {
            
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": f"{prompt}"
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
                "detail": "low" ## low will resize image to 512 x 512 px. high will follow some complicated resizing + divide into 512 x 512 px tokens.
            }
            }
        ]
        }
    ],
    "max_tokens": 2000
    }
    count = 0
    while count < 5:
        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            print(response)
            return response.json()['choices'][0]['message']['content']         
        except:
            time.sleep(5)
            count += 1
    raise('TimeOutError')
   

def main(args):
    # load env variables
    os.chdir('./')
    load_dotenv() 
    # getting OCR keys
    # add API URL and X-OCR-SECRET to .env file
    # Refer to this page: https://data-make.tistory.com/696
    client = OpenAI()
    url = os.getenv('API_URL')
    headerDict = {
        'X-OCR-SECRET': os.getenv('X-OCR-SECRET')
    }
    patent_dicts = list()

    # Load csv file and set index from id
    df = pd.read_csv(args.csv_path)
    df.set_index('id', inplace=True)
    image_dirs = glob(args.image_path+'/*')
    # Iterating over image directories
    for dir in tqdm(image_dirs):
        patent_dict = dict( 
            application_number = dir.split('/')[-1].split('pdf_')[-1], # 출원 번호
            publication_number = '', # 공개 번호
            patent_number = '', # 등록 번호
            drawings = [],
        )
        # extract corresponding '도면의간단한설명' by application number
        extracted_data = df.loc[patent_dict['application_number'], '도면의간단한설명']
        extracted_data = split_text_by_pattern(extracted_data)
        # 이미지를 순서대로 불러와서, -> 고쳐야 함 (p{d}-{d}.png 인데 숫자 부분을 문자로 인식해서 8 보다 10이 먼저 옴)
        images = glob(dir+'/*.png')
        images = sorted(images, key=lambda x: get_sort_key(x))
        # length of images is required to be same with length of extracted data (excluding 대표도)
        # assert len(image/s)-1 == len(extracted_data)
        # 일단 id는 0부터 시작해서 순서대로 먹이고
        for i, img in tqdm(enumerate(images)):
            if i == 0: #  and img.split('/')[-1] == 'p0-6.png': -> 대표도가 없는 경우도 있나? 그렇다면 복잡한 처리 필요.
                continue # skip 대표도
            base64_image = encode_image(img) # encoding image
            # OCR API Call
            ocr_result_korean, ocr_result_numbers = ocr_call(base64_image, url, headerDict)
            # GPT API Call
            gpt_result = gpt_call(ocr_result_korean, ocr_result_numbers, base64_image)
            image_dict = dict(
                drawing_id = i,
                path = img,
                numbers = ocr_result_numbers, # from ocr
                caption = extracted_data[i-1] + gpt_result # add 도면의 간단한 설명 as caption
            )
            patent_dict['drawings'].append(image_dict)
            print("image:", img)
            print("processed result:", image_dict)
        with open(args.json_path, 'a') as f:
            json.dump(patent_dict, f, indent=4, ensure_ascii = False)
            f.write('\n')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--csv_path', type = str, default = '../pdf_process/data.csv',
                        help='path to csv file')
    parser.add_argument('--image_path', type = str, default = './image',
                        help='path to image file')
    parser.add_argument('--json_path', type = str, default = './mock_data.json',
                        help='path to json (saved) file')


    args = parser.parse_args()   
    main(args)