import time
import os
import random

from tqdm import tqdm
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException

def crawl_registration_number(driver: webdriver, patent_id: str) -> str:
    """
    @param driver: selenium webdriver
    @param patent_id: 14 digit integer passed as a string. either publication number (등록번호) or patent number (공개번호)
    @param id_type: one of the values ["publication_number", "patent_number"]
    
    @returns registration number string
    """
        
    driver.get('http://kpat.kipris.or.kr/kpat/searchLogina.do?next=MainSearch')
    wait = WebDriverWait(driver, 30)
    action = ActionChains(driver)
    
    # time.sleep(5) # Let the user actually see something!
    
    ## Type patent_id to the appropriate search box according to id_type
    if patent_id[-4:] == "0000":
        patent_id_for_search = f"GN=[{patent_id}]"
    else:
        patent_id_for_search = f"OPN=[{patent_id}]"
    
    # print(patent_id_for_search)
    search_box = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '#queryText')))
    action.move_to_element(search_box).click().send_keys(patent_id_for_search).send_keys(Keys.RETURN).perform()    
    
    try:
        # Assert that there's only one search result
        result_count = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '#divMainArticle > form > section > div.float_left > p > span.total')))
        result_count = int(result_count.text)
        assert result_count == 1, "More than one search result"
        
        # get registration number
        registration_number = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '#mainsearch_info_list > div.mainlist_topinfo > li:nth-child(3) > span.point01 > a')))
        registration_number = registration_number.text
        registration_number = registration_number.strip().split()[0]
                
        return registration_number
        
    except Exception as e:
        
        print(f"Skipping {patent_id}...")
        # print(f"Error message: {e}")        
        return ""
    

if __name__ == "__main__":
    
    ## set path to patent_search repository
    os.chdir('/Users/hayley/Documents/p4ds/patent_search') #CHANGE THIS LINE
    # os.chdir('/home/hyeryung/patent_search')
    
    ## read data
    patent_data = pd.read_csv('kipris_plus/prior_arts_formatted.csv')
    # patent_data = patent_data.loc[patent_data['exists']==0].copy()
    patent_data['Prior출원번호'] = ''
    
    ## declare selenium webdriver
    options = Options()
    options.add_argument("--headless=new")
    test_driver = webdriver.Chrome(options=options)#'/path/to/driver'  # Optional argument, if not specified will search path.
    
    
    if os.path.exists('kipris_plus/output.txt'):
        with open('kipris_plus/output.txt', 'r') as f:
            already_crawled_data = f.readlines()
        print("file already exists...")
        start_index = len(already_crawled_data)
        f = open('kipris_plus/output.txt', 'a')
    else:
        start_index = 0
        f = open('kipris_plus/output.txt', 'w')
    
    print(f"start_index: {start_index}")
    try:
        for i, row in tqdm(patent_data.iloc[start_index:,:].iterrows()):
            
            reg_number = crawl_registration_number(test_driver, str(row['일련번호']))
            f.write(f"{row['출원번호']},{reg_number}\n")
            f.flush()
        
            time.sleep(0.5+random.randint(0,1))
        

        test_driver.quit()
        f.close()
    except:
        f.close()
        