## How to run this code
- step 1. 'pip install -r requirements.txt' in your terminal.
- step 2. Download the pdf resources in folder 'pdf_sources'
- step 3. 'pdf_process'라는 폴더가 메인이 되게끔 파이썬을 여세요.
- step 4. Just run 'main.py'.

## How to download data
Run `download_data.sh` in `pdf_process` directory.

### Notes to users
- data.csv에 추출된 파일들이 있습니다. 전처리 방식이 변경되면 data.csv도 바뀌겠지만, 컬럼의 형태는 안 바뀌고 본문만 바뀔 것 같아요.
- excel로 열면 한글이 다 깨져서 보이네요. row 개수도 이상하게 늘어나있어요.
- 근데 파이썬으로 열면 잘 보여요. 모든게 다 정상적으로.
- image file들은 image라는 폴더에 있습니다. image/{pdf파일이름}/{이미지 이름}.png 의 형태입니다.

### Notes for data processing 담당자
- 전처리 작업한 파일은 zip 해서 drive에 올리신 후에 `download_data.sh` 파일에 경로를 업데이트 해주세요.
- 파일 링크 (가령 `https://drive.google.com/drive/folders/1WMlY2QYlIVvBbYYNt4X7fHMJuCYXKhI5?usp=sharing`) 에서 `folders/`와 `?usp=sharing` 사이에 있는 string (가령 `1WMlY2QYlIVvBbYYNt4X7fHMJuCYXKhI5`) 을 업데이트 해주시면 됩니다.
