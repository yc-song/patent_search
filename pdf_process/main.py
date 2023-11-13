import os, shutil
import pandas as pd
from utils import from_pdf_to_images
from utils import from_pdf_to_html
from utils import from_html_to_df

csv_path = 'data.csv' ## 최종적으로 생산될 csv 파일의 이름
resource_pdf_folderName= 'pdf_sources' ## 크롤링한 pdf 파일을 모두 이 폴더에 담으면 됨

pdf_files = [f for f in os.listdir(resource_pdf_folderName) if f.endswith('.pdf')]
pdf_names = [s[:-4] for s in pdf_files]
df_list = []

for i, (name, file) in enumerate(zip(pdf_names, pdf_files)):
    image_path = os.path.join(os.getcwd(), "image", "pdf_"+name)
    html_path = os.path.join(os.getcwd(), "html", f"{name}.html")
    from_pdf_to_html(os.path.join(resource_pdf_folderName, file), html_path)
    from_pdf_to_images(os.path.join(resource_pdf_folderName, file),image_path, include_cls_image = True, initialize_folder=True)
    df = from_html_to_df(html_path, name)
    df_list.append(df)
    print(f"processing.... {i+1}/{len(pdf_files)} is done.")

if os.path.exists(csv_path):
    os.remove(csv_path)

final_df = pd.concat(df_list, ignore_index = True)
final_df.to_csv(csv_path, index = False, encoding= 'utf-8')

# if os.path.exists('html'):
#     shutil.rmtree('html')