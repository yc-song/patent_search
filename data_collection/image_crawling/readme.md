This is a script to crawl images from google patents based on patent numbers.

# Setup
- I used python 3.9.
- For selenium, 
    - download chromedriver https://googlechromelabs.github.io/chrome-for-testing/. 
    - place chromedriver exe file in `/usr/local/bin`
        - Or create a symbolic link (c.f. `sudo ln -s /Users/hayley/Documents/chromedriver-mac-arm64/chromedriver /usr/local/bin/chromedriver`)
- Set up google cloud cli.
    1. install:  https://cloud.google.com/sdk/docs/install
    2. initialize: https://cloud.google.com/sdk/docs/initializing
    3. authorize: https://cloud.google.com/docs/authentication/provide-credentials-adc#local-dev
    4. run this `gcloud auth application-default login `
- Finally, run this  `pip install -r requirements.txt`

# Run
1. Prepare a list of patent numbers `python prepare_csv.py`
2. Change the list of patent numbers directory in main.py and run `python main.py`

# Info
23.11.19
- Number of patents: 152
- List of patent numbers: `data_collection/patent_numbers_for_crawling.csv`
    - Patent numbers b/c in google patents you can only search with patent numbers.
- Image crawled are uploaded to [google cloud storage](https://console.cloud.google.com/storage/browser/kipris/KI202311161011280001/crawled_images?authuser=2&hl=en&project=p4ds-team-2-2023fall&pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false)
    - Note: patents {'102587216', '102595326'} have no images b/c they have no images.