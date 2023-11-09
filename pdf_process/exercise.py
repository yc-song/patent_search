## 그냥 연습장~

import os, getpass, shutil
from utils import patent_HTMLParser

with open('pdf_2/2.html') as f:
    parser = patent_HTMLParser()
    parser.feed(f.read())
    
for key, val in parser.contents.items():
    print(f"key: {key}")
    print(f"val: {val}")
    print()