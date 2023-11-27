import pandas as pd
import os

def format_output_txt_to_kipris_format(inpath:str="kipris_plus/outputs/output.txt",
                                       outpath:str = "kipris_plus/outputs/prior_arts_for_crawling.txt") -> None:
    
    """
    read output.txt from crawl_registration_number.py and then..
    drop duplicates, drop patents already crawled, and then save as kipris_plus format.
    """
    
    labels = pd.read_csv(inpath,dtype=str,names=['', 'target_regist_number'])
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

    targets.to_csv(outpath, index=False, header=False)
    

if __name__ == "__main__":

    os.chdir('/Users/hayley/Documents/p4ds/patent_search')

    format_output_txt_to_kipris_format()
    