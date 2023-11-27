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

def crawl_one_patent(driver: webdriver, patent_id: str, out_path: str = "crawled_images") -> None:
    
    if len(patent_id) > 9:
        patent_id = patent_id[:9]
        
    os.makedirs(os.path.join(out_path, patent_id), exist_ok=True)
    
    patent_id_for_search = f"KR{patent_id}"
        
    driver.get(f'https://patents.google.com/patent/{patent_id_for_search}')
    wait = WebDriverWait(driver, 30)
    action = ActionChains(driver)
    
    # time.sleep(5) # Let the user actually see something!
    
    # # Perform Search
    # search_box = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '#searchInput')))
    # action.move_to_element(search_box).send_keys(patent_id_for_search).send_keys(Keys.RETURN).perform()
    
    try:
        # Get image count
        image_count = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '#thumbnails > h3 > span')))
        image_count = int(image_count.text)
        
        print(f"{patent_id_for_search}: crawling {image_count} images.")
        
        for i in range(1, image_count+1):
            
            # select image thumbnail to get larger image
            image_box = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, f'#figures > div > img:nth-child({i})')))
            action.move_to_element(image_box).click().perform()
            
            parent = driver.current_window_handle
            
            # open in another bigger window 
            button = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '#wrap > div.header.layout.horizontal.style-scope.image-viewer > div > span:nth-child(5) > a > paper-icon-button')))
            action.move_to_element(button).click().perform()
            
            allWindows = driver.window_handles
            for winId in allWindows:
                if winId != parent: 
                    # switch to popped up window
                    driver.switch_to.window(winId)
                    
                    # select large image and download
                    img = None
                    while img is None:
                        try:
                            img = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'body > img')))
                        except WebDriverException:
                            print('retrying...')
                            pass
                        
                    # download the image
                    with open(os.path.join(out_path, patent_id, f"img{i}.png"), 'wb') as f:
                        f.write(img.screenshot_as_png)
                    
                    time.sleep(0.5)
                    
                    # close the window
                    driver.close()
            # switch back to current window
            driver.switch_to.window(parent)
            
            # close the pop up window and go back to the main window
            time.sleep(0.5)
    except:
        print(f"Skipping {patent_id}...")        

    

if __name__ == "__main__":
    
    ## set path to patent_search repository
    os.chdir('/Users/hayley/Documents/p4ds/patent_search') #CHANGE THIS LINE
    
    ## read data
    patent_data = pd.read_csv('data_collection/extracted_data_formatted.csv', index_col=0, dtype=str) ## CHANGE THIS LINE
    patent_data = patent_data.loc[~patent_data['등록번호'].isna(),:].copy()
    patent_list = patent_data['등록번호'].tolist()
    print(len(patent_list))
    
    options = Options()
    # options.add_argument("--headless=new")
    test_driver = webdriver.Chrome(options=options)#'/path/to/driver'  # Optional argument, if not specified will search path.
    
    os.makedirs('data_collection/crawled_images', exist_ok=True)
    already_crawled_list = os.listdir('data_collection/crawled_images')
    found_count = 0
    
    for patent_number in tqdm(patent_list):
        
        if patent_number[:9] in already_crawled_list:
            found_count += 1
            continue
        crawl_one_patent(test_driver, patent_number, "data_collection/crawled_images")
        time.sleep(random.randint(0,2))
    
    test_driver.quit()
    
    print(f"{found_count} patents skipped since already crawled.")
