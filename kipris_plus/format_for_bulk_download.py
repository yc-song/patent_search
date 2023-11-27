import pandas as pd
import os
os.chdir('/Users/hayley/Documents/p4ds/patent_search')

labels = pd.read_csv('kipris_plus/output.txt',dtype=str,names=['', 'target_regist_number'])
print(labels.head())
targets = labels[['target_regist_number']].copy()
print(f"Num. target prior arts before drop dup: {targets.shape[0]}")

## drop duplicates
targets = targets.drop_duplicates()
print(f"Num. unique target prior arts: {targets.shape[0]}")

## drop patents that have already been crawled
already_crawled_data = pd.read_csv('data_preprocess/extracted_data_formatted_merged.csv',dtype=str)
print(f"Num. target prior arts already crawled: {targets.loc[targets['target_regist_number'].isin(already_crawled_data['출원번호'].tolist())].shape[0]}")

targets = targets.loc[~targets['target_regist_number'].isin(already_crawled_data['출원번호'].tolist())].copy()
print(f"Final Num. target prior arts to crawl: {targets.shape[0]}")

targets.to_csv("kipris_plus/additional_patents.txt", index=False, header=False)