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

def crawl_registration_number(driver: webdriver, patent_id: str, id_type: str) -> str:
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
    if id_type == "publication_number":
        patent_id_for_search = f"OPN=[{patent_id}]"
    elif id_type == "patent_number":
        # if len(patent_id) > 9:
        #     patent_id = patent_id[:9]
        patent_id_for_search = f"GN=[{patent_id}]"
    
    # print(patent_id_for_search)
    search_box = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '#queryText')))
    action.move_to_element(search_box).click().send_keys(patent_id_for_search).send_keys(Keys.RETURN).perform()    
    
    # search_button = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '#btnItemizedSearch')))
    # action.move_to_element(search_button).click().perform()
    
    try:
        # Assert that there's only one search result
        result_count = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '#divMainArticle > form > section > div.float_left > p > span.total')))
        result_count = int(result_count.text)
        assert result_count == 1, "More than one search result"
        
        # Save current window handle (for later)
        parent = driver.current_window_handle
        
        # get registration number
        registration_number = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '#mainsearch_info_list > div.mainlist_topinfo > li:nth-child(3) > span.point01 > a')))
        registration_number = registration_number.text
        registration_number = registration_number.strip().split()[0]
                
        
        # # Click the search result
        # search_result = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, f'#divViewSel{registration_number} > div.search_section_title > h1 > a:nth-child(2)')))
        # action.move_to_element(search_result).click().perform()
        # #divViewSel2019940032136 > div.search_section_title > h1 > a:nth-child(2)
        # #mainsearch_info_list > div.mainlist_topinfo > li:nth-child(3) > span.point01 > a
        # allWindows = driver.window_handles
        # for winId in allWindows:
        #     if winId != parent: 
        #         # switch to popped up window
        #         driver.switch_to.window(winId)
                
        #         # get patent_number
        #         patent_number = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '#divBiblioContent > div.detial_plan_info > ul > li:nth-child(5)')))
        #         patent_number = patent_number.text
        #         patent_number = patent_number.strip().split()[0]
                
        #         # # get registration number
        #         # registration_number = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '#divBiblioContent > div.detial_plan_info > ul > li:nth-child(3)')))
        #         # registration_number = registration_number.text
        #         # registration_number = registration_number.strip().split()[0]
                
        #         # get publication number
        #         publication_number = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '#divBiblioContent > div.detial_plan_info > ul > li:nth-child(6)')))
        #         publication_number = publication_number.text
        #         publication_number = publication_number.strip().split()[0]
                
        #         time.sleep(0.5)
                
        #         # close the window
        #         driver.close()
        # # switch back to current window
        # driver.switch_to.window(parent)
        
        # time.sleep(0.5)
        
        # return {"registration_number": registration_number, 
        #         "publication_number": publication_number,
        #         "patent_number": patent_number}
        return registration_number
        
    except Exception as e:
        
        print(f"Skipping {patent_id}...")
        print(f"Error message: {e}")        
        return ""
    

if __name__ == "__main__":
    
    ## set path to patent_search repository
    os.chdir('/Users/hayley/Documents/p4ds/patent_search') #CHANGE THIS LINE
    # os.chdir('/home/hyeryung/patent_search')
    
    ## read data
    patent_data = pd.read_csv('kipris_plus/prior_arts_formatted.csv')
    patent_data = patent_data.loc[patent_data['exists']==0].copy()
    patent_data['Prior출원번호'] = ''
    
    ## declare selenium webdriver
    options = Options()
    options.add_argument("--headless=new")
    test_driver = webdriver.Chrome(options=options)#'/path/to/driver'  # Optional argument, if not specified will search path.
    
    for i, row in tqdm(patent_data.iterrows()):
        
        if row['특허유형'] in ['A', 'U']:
            reg_number = crawl_registration_number(test_driver, str(row['일련번호']), "publication_number")
        elif row['특허유형'] in ['B1', 'Y1']:
            reg_number = crawl_registration_number(test_driver, str(row['일련번호']), "patent_number")    
        patent_data.iloc[i, -1] = reg_number
    
        time.sleep(random.randint(0,2))
    

    test_driver.quit()
    
    patent_data.to_csv('result.csv',index=False)
    