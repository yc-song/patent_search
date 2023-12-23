## 그냥 연습장~


import os, shutil
import pandas as pd

from utils import patent_HTMLParser
if os.path.exists('html'):
    shutil.rmtree('html')