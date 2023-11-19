import re
import io
import os
import logging

import pandas as pd
from google.cloud import storage

logging.basicConfig(level=logging.DEBUG, format='%(message)s')
logger = logging.getLogger(__file__)
logger.setLevel(os.environ.get("LOGGING_LEVEL", "DEBUG"))

def list_blobs_by_prefix_regex(bucket_name,  prefix=None, regex_exp=".*"):
    """Lists all the blobs in the bucket."""
    # bucket_name = "your-bucket-name"

    storage_client = storage.Client()

    # Note: Client.list_blobs requires at least package version 1.17.0.
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix)
    
    regex = re.compile(regex_exp)
    blobs = [x for x in blobs if re.search(regex, x.name) is not None ]

    return blobs
        

if __name__ == "__main__":
    os.chdir('/Users/hayley/Documents/p4ds/patent_search')

    bibfiles = list_blobs_by_prefix_regex("kipris", 
                                          regex_exp='KI202311161011280001/crawled_images/\d+/', 
                                          prefix='KI202311161011280001/crawled_images/')
    # print(bibfiles[:10])
    bibfiles = [x.name.split('/')[-2] for x in bibfiles]
    bibfiles = list(set(bibfiles))
    # print(bibfiles[:10])

    already_crawled_list = os.listdir('data_collection/crawled_images')
    
    print("List of patents not uploaded:")
    print(set(already_crawled_list) - set(bibfiles))
    
    # 이미지가 없는 특허들임
    # List of patents not uploaded:
    # {'102587216', '102595326'}
