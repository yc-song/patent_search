import requests
import xmltodict
import pandas as pd
import requests
import time

def get_response(url):
    while True:
        try:
            response = requests.get(url)
            time.sleep(1)
            break
        except:
            print("error!!")
            time.sleep(60*15)
    content = response.text
    content = xmltodict.parse(content)['response']
    return content


def extract_items_from_body(content):
    result = content['body']['item']['path']
    return result


def main():
    api_key = 'put your api'
    patent_number = 1020050050026
    url = 'http://plus.kipris.or.kr/kipo-api/kipi/patUtiModInfoSearchSevice/getPubFullTextInfoSearch?applicationNumber={}&ServiceKey={}'.format(patent_number, api_key)
    content = get_response(url)
    result_link = extract_items_from_body(content)
    pdf_response = requests.get(result_link)
    with open("./pdf/{}.pdf".format(patent_number), "wb") as pdf_file:
        pdf_file.write(pdf_response.content)


if __name__ == '__main__':
    main()