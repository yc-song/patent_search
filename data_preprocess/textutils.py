import re
import math
from collections import defaultdict
from typing import Union, List
import json

import pandas as pd

def extract_ref_from_text(x: Union[str, float]) -> List:
    init_extract = list(set(re.findall(r"(?<=\()\d*[a-zA-Z]?(?=\))|도 \d+", str(x))))
    if len(init_extract) > 0:
        return init_extract
    else:
        return []
    

def get_number2drawing_dict(data_json_path: str) -> dict:
    
    patent_drawing_json = json.load(open(data_json_path, 'r'))
    patent_drawing_df = pd.DataFrame(patent_drawing_json)
    
    def convert_draw2num_to_num2draw(x):
        
        num2drawing = defaultdict(list)
        for drawing in x:
            for num in drawing['numbers']:
                num2drawing[num].append(drawing['drawing_id'])
        return num2drawing
    
    patent_drawing_df['num2drawing'] = patent_drawing_df['drawings'].apply(convert_draw2num_to_num2draw)
    patent_drawing_df.set_index('application_number', inplace=True)
    patent_drawing_dict = patent_drawing_df.to_dict(orient='index')
    return patent_drawing_dict



# def convert_ref_to_drawing_num(x, colname: str, _image_df: pd.DataFrame):
#     drawing_ids = []
#     if type(x[f"{colname}_ref"]) != float:
#         for ref in x[f"{colname}_ref"]:
#             if '도' in ref:
#                 drawing_ids.append(int(ref.strip('도').strip()))
#             else:
#                 num2drawing_dict = _image_df.loc[_image_df['application_number'].astype(str) == str(x['id']),'num2drawing'].values[0]
#                 drawing_ids.extend(num2drawing_dict[ref])
#     return list(set(drawing_ids))

    
def convert_refs_to_drawing_num(refs: List, num2drawing_dict: dict) -> List:
    
    drawing_ids = []
    for ref in refs:
        if '도' in ref:
            drawing_ids.append(int(ref.strip('도').strip()))
        else:
            print(ref, '--->', num2drawing_dict.get(ref, []))
            drawing_ids.extend(num2drawing_dict.get(ref, []))
    return list(set(drawing_ids))
    

# def extract_description_for_image(x: Union[str, float]):
    
#     descs = re.findall(r'도 ?\d+[의은는][^\.]+\.', str(x))
    
#     return dict(zip([f'도 {i}' for i in range(1, len(descs)+1)], descs))

# def extract_description_for_code(x):
#     ids = re.findall(r'([\d]+)[a-zA-Z]?,? ?[\da-zA-Z]*(?= ?[:;]+)', str(x)) # became ugly to take care of cases like '311b, 311c'
#     ids = [x.strip() for x in ids]
#     descs = re.findall(r'(?<=[:;])[\-0-9ㄱ-ㅎ가-힣a-zA-Z ]+', str(x))
#     descs = [re.sub(r'([\d]+)[a-zA-Z]?$', '', x.strip()) for x in descs]
#     descs = [x.strip() for x in descs]
#     if len(descs)>0: # handling noise from data parsing ('도면' got attached to '부호의설명' column)
#         descs[-1] = re.sub('도면.+', '', descs[-1]).strip()
        
#     return_dict = dict(zip(ids, descs))
    
#     initial_keys = list(return_dict.keys()) # if there're keys like '311b, 311c', split the key into two different keys and add the same value.
#     for key in initial_keys:
#         if ',' in key:
#             value = return_dict[key]
#             return_dict.pop(key)
#             subkey_list = key.split(',')
#             for subkey in subkey_list:
#                 return_dict[subkey.strip()] = value
    
#     return return_dict