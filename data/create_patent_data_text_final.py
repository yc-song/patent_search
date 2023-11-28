#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import os

def main():
    # data = pd.read_csv('/Users/hayley/Documents/p4ds/patent_search/data_preprocess/extracted_data_formatted_merged.csv',index_col=0, dtype=str)
    # print(data.drop_duplicates().shape)
    # data1 = pd.read_csv('/Users/hayley/Documents/p4ds/patent_search/data_collection/extracted_data_formatted.csv', dtype=str)
    # # print(data.columns == data1.columns)
    # print(data1.drop_duplicates().shape)

    # data1['출원번호']=data1['출원번호'].str.strip()
    # data['출원번호']=data['출원번호'].str.strip()

    # data['query'] = 1
    # data1['query'] = 0

    # data_all = pd.concat([data, data1],axis=0)

    # print(data_all.shape, data_all.drop_duplicates(subset=['출원번호']).shape)

    # data_all.to_csv("data/patent_data_text.csv",index=False)



    # ### Indicate whether image exists

    data_all = pd.read_csv("data/patent_data_text.csv",dtype=str)
    print(data_all['query'].value_counts())
    data_all['query'] = data_all['query'].astype(int)
    
    imgs = os.listdir('data/images')
    gold_imgs = os.listdir('data/gold_images')
    data_all['image_yn'] = 0
    data_all.loc[data_all['등록번호'].fillna('').str[:9].isin(imgs),'image_yn'] = 1
    data_all.loc[data_all['등록번호'].fillna('').str[:9].isin(gold_imgs),'image_yn'] = 1
    print(data_all['image_yn'].value_counts())

    # ### Indicate whether prior-arts exists
    labels = pd.read_csv('data/labels.csv',dtype=str)
    data_all['labelled_yn'] = 0
    data_all.loc[data_all['출원번호'].isin(labels['source']),'labelled_yn']=1
    print(data_all['labelled_yn'].value_counts())
    
    data_all['target'] = 0
    data_all.loc[data_all['출원번호'].isin(labels['target']),'target']=1
    print(data_all.target.value_counts())
    
    # ### Summary of counts
    data_all['purely_query'] = 0
    data_all.loc[(data_all['query']==1)&(data_all['target']!=1),'purely_query']=1
    print(data_all['purely_query'].value_counts())
    
    data_all['labelled_yn_query']=0
    data_all.loc[(data_all['labelled_yn']==1)&(data_all['purely_query']==1),'labelled_yn_query']=1
    print(data_all['labelled_yn_query'].value_counts())
    
    data_all.groupby(['query', 'target', 'labelled_yn', 'image_yn']).size().to_csv('data/data_statistics.csv')
    # data_all.groupby(['purely_query', 'labelled_yn_query', 'image_yn']).size()
    # data_all.groupby(['purely_query', 'labelled_yn_query', 'image_yn']).size().to_csv('data/data_statistics.csv')
    data_all.to_csv('data/patent_data_text_final.csv',index=False)


if __name__ == "__main__":
    os.chdir("/Users/hayley/Documents/p4ds/patent_search")
    main()




