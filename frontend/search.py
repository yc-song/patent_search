# ##########################################################
# # to run: streamlit run main.py
# ##########################################################
import requests
import streamlit as st
from datetime import datetime

import os  # 경로 탐색

url = "http://localhost:5000"
# 파일 업로드 함수
# 디렉토리 이름, 파일을 주면 해당 디렉토리에 파일을 저장해주는 함수
def save_uploaded_file(directory, file):
    # 1. 저장할 디렉토리(폴더) 있는지 확인
    #   없다면 디렉토리를 먼저 만든다.
    if not os.path.exists(directory):
        os.makedirs(directory)

    # 2. 디렉토리가 있으니, 파일 저장
    with open(os.path.join(directory, file.name), 'wb') as f:
        f.write(file.getbuffer())
    return st.success('파일 업로드 성공!')


# 기본 형식
def main():
    st.title('Patent Search Engine')

    st.subheader('We support the following:')
    st.markdown("1. Searching for similar patents through drawings.\n"
                "2. Patent search based on descriptions or keywords.\n"
                "3. A combination of 1 and 2.")
    img_file = st.file_uploader('Please upload an image.', type=['png', 'jpg', 'jpeg'])
    query = 0
    query = st.text_input('Type your question')
    image_name = 0

    if st.button('send'):
        if img_file is not None:  # 파일이 없는 경우는 실행 하지 않음
            print(type(img_file))
            print(img_file.name)
            print(img_file.size)
            print(img_file.type)

            # 유저가 올린 파일을,
            # 서버에서 처리하기 위해서(유니크하게)
            # 파일명을 현재 시간 조합으로 만든다.
            current_time = datetime.now()
            print(current_time)
            print(current_time.isoformat().replace(':', "_") + '.jpg')  # 문자열로 만들어 달라
            # 파일 명에 특정 특수문자가 들어가면 만들수 없다.
            filename = current_time.isoformat().replace(':', "_") + '.jpg'
            img_file.name = filename
            image_name = f'../frontend/image/{img_file.name}'

            save_uploaded_file('image', img_file)

        output = requests.get(f"{url}/api/data",
                              params={"image_name": image_name, "query": query}).json()
        st.header("Results")
        for output in output['out']:
            st.markdown(output['summary'])
            st.image(output['image'])


if __name__ == '__main__':
    main()