import pandas as pd
import io

# Sample data provided
txt_file_path = './from_hr/DBII_000000000000003/TXT/PriorTechnologyDocument.txt'
output_csv_path = './prior_arts_formatted.csv'
output_xlsx_path = './prior_arts_formatted.xlsx'

# Load data into a pandas DataFrame
df = pd.read_csv(txt_file_path, delimiter='¶')


# Dropping unnecessary columns

# Filtering rows with '선행기술조사문헌번호' starting with 'KR' and removing unwanted patterns
df[['일련번호', '특허유형']] = df['선행기술조사문헌번호'].str.split(' ', n=1, expand=True)
df.drop(['선행기술조사문헌번호','선행기술조사문헌일련번호', '심사관인용여부'], axis=1, inplace=True)
df = df[df['일련번호'].str.startswith('KR')]
df['일련번호'] = df['일련번호'].str.replace('KR', '').str.replace('-', '').str.replace(' ','')
# Padding numbers to meet the length requirement
df['일련번호'] = df['일련번호'].apply(lambda x: x + '0000' if len(x) == 9 else x)
print(df)
df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
df.to_excel(output_xlsx_path)
