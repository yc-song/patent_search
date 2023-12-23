"""
Failed attempt to crawl images from Kipris.
"""

# import time
# import os
# import random

# from tqdm import tqdm
# import pandas as pd
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium import webdriver
# from selenium.webdriver import ActionChains
# from selenium.webdriver.common.by import By
# from selenium.webdriver.common.keys import Keys
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.webdriver.chrome.options import Options
# from selenium.common.exceptions import WebDriverException

# def crawl_patent_image_kipris(driver: webdriver, patent_id: str, id_type: str, out_path: str = "crawled_images_kipris") -> str:
#     """
#     @param driver: selenium webdriver
#     @param patent_id: 14 digit integer passed as a string. either publication number (등록번호) or patent number (공개번호)
#     @param id_type: one of the values ["publication_number", "patent_number"]
    
#     @returns registration number string
#     """
        
#     driver.get('http://kpat.kipris.or.kr/kpat/searchLogina.do?next=MainSearch')
#     wait = WebDriverWait(driver, 30)
#     action = ActionChains(driver)
    

#     # time.sleep(5) # Let the user actually see something!
#     ## Type patent_id to the appropriate search box according to id_type
#     if id_type == "publication_number":
#         patent_id_for_search = f"OPN=[{patent_id}]"
#     elif id_type == "patent_number":
#         patent_id_for_search = f"GN=[{patent_id}]"
#     elif id_type == "registration_number":
#         patent_id_for_search = f"AN=[{patent_id}]"
    
#     search_box = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '#queryText')))
#     action.move_to_element(search_box).click().send_keys(patent_id_for_search).send_keys(Keys.RETURN).perform()    
    
#     # create directory to save
#     if len(patent_id) > 9:
#         patent_id = patent_id[:9]
#     os.makedirs(os.path.join(out_path, patent_id_for_search.replace('=','_').replace('[', '').replace(']','')), exist_ok=True)
    
#     try:
#         # Assert that there's only one search result
#         result_count = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '#divMainArticle > form > section > div.float_left > p > span.total')))
#         result_count = int(result_count.text)
#         assert result_count == 1, "More than one search result"
        
#         # Save current window handle (for later)
#         parent = driver.current_window_handle
        
#         # get registration number
#         registration_number = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '#mainsearch_info_list > div.mainlist_topinfo > li:nth-child(3) > span.point01 > a')))
#         registration_number = registration_number.text
#         registration_number = registration_number.strip().split()[0]
                
#         # Click the search result
#         search_result = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, f'#divViewSel{registration_number} > div.search_section_title > h1 > a:nth-child(2)')))
#         action.move_to_element(search_result).click().perform()
        
#         allWindows = driver.window_handles
#         for winId in allWindows:
#             if winId != parent: 
#                 # switch to popped up window
#                 driver.switch_to.window(winId)
                
#                 currWindow = driver.current_window_handle
                
#                 # click drawing checkbox
#                 checkbox = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '#allImg')))
#                 action.move_to_element(checkbox).click().perform()
                
#                 # go to the all imagebox list
#                 # html_list = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, '#allImgList')))
#                 # num_items = len(html_list.find_elements_by_tag_name("li"))
#                 print('A')
#                 html_list = driver.find_element(By.ID, "allImgList")
#                 print(html_list)
#                 num_items = len(html_list.find_elements(By.TAG_NAME, "li"))
#                 print('C')
                
#                 print(num_items)
#                 for i in range(0, num_items):
                    
#                     # select large image and download
#                     img = None
#                     while img is None:
#                         try:
#                             img = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, f'#img_{i}')))
#                         except WebDriverException:
#                             print('retrying...')
#                             pass
                        
#                     # download the image
#                     with open(os.path.join(out_path, patent_id_for_search.replace('=','_').replace('[', '').replace(']',''), f"img{i}.png"), 'wb') as f:
#                         f.write(img.screenshot_as_png)
                    
#                     time.sleep(0.5)
                    
#                 # close the popup window
#                 driver.close()
#         # switch back to current window
#         driver.switch_to.window(parent)
#         time.sleep(0.5)
        
        
#     except Exception as e:
        
#         print(f"Skipping {patent_id}...")
#         print(f"Error message: {e}")        
    

# if __name__ == "__main__":
    
#     ## set path to patent_search repository
#     os.chdir('/Users/hayley/Documents/p4ds/patent_search') #CHANGE THIS LINE
#     # os.chdir('/home/hyeryung/patent_search')
    
#     ## read data
#     patent_data = pd.read_csv('data_preprocess/extracted_data_formatted_merged.csv', index_col=0, dtype=str)
#     patent_list = patent_data['등록번호'].tolist()
#     print(len(patent_list))
    
#     options = Options()
#     options.add_argument("--headless=new")
#     test_driver = webdriver.Chrome(options=options)#'/path/to/driver'  # Optional argument, if not specified will search path.
    
#     os.makedirs('data_collection/crawled_images_kipris', exist_ok=True)
#     already_crawled_list = os.listdir('data_collection/crawled_images_kipris')
#     found_count = 0
    
#     my_num = 0  # CHANGE THIS LINE : change it to number next to your name # hyeryung: 0, jong: 1, mooho: 2, moonwon: 3
#     num_len = 478
#     start_index = my_num * num_len
#     end_index = (my_num + 1) * num_len
#     for patent_number in tqdm(patent_list[start_index:end_index]):
        
#         if patent_number[:9] in already_crawled_list:
#             found_count += 1
#             continue
#         crawl_patent_image_kipris(test_driver, patent_number, "patent_number", "data_collection/crawled_images_kipris")
#         time.sleep(random.randint(0,2))
    
#     test_driver.quit()
    
#     print(f"{found_count} patents skipped since already crawled.")
