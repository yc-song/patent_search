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
        
def download_blob_to_stream(bucket_name, source_blob_name, file_obj):
    """Downloads a blob to a stream or other file-like object."""

    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"

    # The ID of your GCS object (blob)
    # source_blob_name = "storage-object-name"

    # The stream or file (file-like object) to which the blob will be written
    # import io
    # file_obj = io.BytesIO()

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    # Construct a client-side representation of a blob.
    # Note `Bucket.blob` differs from `Bucket.get_blob` in that it doesn't
    # retrieve metadata from Google Cloud Storage. As we don't use metadata in
    # this example, using `Bucket.blob` is preferred here.
    blob = bucket.blob(source_blob_name)
    blob.download_to_file(file_obj)

    # print(f"Downloaded blob {source_blob_name} to file-like object.")

    return file_obj


if __name__ == "__main__":
    os.chdir('/Users/hayley/Documents/p4ds/patent_search')

    
    
    bibfiles = list_blobs_by_prefix_regex("kipris", regex_exp='KI.*/.*/TXT/Bibliographic\.txt')
    # print(bibfiles)

    file_obj = io.BytesIO()

    patent_numbers = set()
    for bib in bibfiles:
        
        logger.debug(bib)
        return_obj = download_blob_to_stream("kipris", bib.name, file_obj)
        return_obj.seek(0)
        df = pd.read_csv(return_obj, sep='¶' , dtype=str, engine="python")
        
        logger.debug(df.등록번호)
        patent_numbers |= set(df.등록번호.values.tolist())
        
    patent_numbers = list(patent_numbers)
    patent_numbers = [x for x in patent_numbers if re.match('\d+', x) is not None]
    logger.debug(f"patent_numbers count: {len(patent_numbers)}")
    logger.debug(f"patent_numbers object: {patent_numbers}")
    with open('data_collection/patent_numbers_for_crawling.csv', 'w') as f:
        f.writelines('\n'.join(patent_numbers))
