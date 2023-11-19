import pandas as pd
import re
import numpy as np
import openpyxl
def extract_text(tag, text):
    """Extract text for a given tag from the provided text."""
    pattern = f'<{tag}>(.*?)</{tag}>'
    remove_pattern = r"\<.*?\>"
    match = re.search(pattern, text, re.DOTALL)
    return re.sub(remove_pattern, "", match.group(1).strip()) if match else ''

def delete_tag(text):
    remove_pattern = f"\<.*?\>"
    return re.sub(remove_pattern, "", text)


# Define the tags
tags_and_columns = {
    'invention-title': '발명의 명칭',
    'technical-field': '기술분야',
    'background-art': '배경기술',
    'citation-list': '선행기술문헌',
    'summary-of-invention': '발명의 내용',
    'tech-problem': '해결하려는 과제',
    'tech-solution': '과제의 해결 수단',
    'advantageous-effects': '발명의 효과',
    'description-of-drawings': '도면의 간단한 설명',
    'description-of-embodiments': '발명을 실시하기 위한 구체적인 내용'
}

# Read the content of the text file
file_path = './DBII_000000000000003/TXT/Specification.txt'
entries = []  # List to store each entry's data

with open(file_path, 'r', encoding='utf-8') as file:
    for i, line in enumerate(file):
        ## Process PCT cases (e.g. 1020087013681)
        if 'PCTDisclosure' in line:
            line = line.replace('PCTInventionTitle', 'invention-title')
            line = line.replace('PCTTechnicalField', 'technical-field')
            line = line.replace('PCTBackgroundArt', 'background-art')
            line = line.replace('PCTDisclosure', 'summary-of-invention')
            line = line.replace('PCTDescriptionDrawings', 'description-of-drawings')
            line = line.replace('PCTExample', 'description-of-embodiments')
        elif 'Disclosure' in line:
            line = line.replace('InventionTitle', 'invention-title')
            line = line.replace('TechnicalField', 'technical-field')
            line = line.replace('BackgroundTech', 'background-art')
            line = line.replace('SolutionProblem', 'tech-problem')
            line = line.replace('MeansProblemSolution', 'tech-solution')
            line = line.replace('Effectiveness', 'advantageous-effects')
            line = line.replace('InventDetailContent', 'InventDetailContent')
        if i> 0:
            application_number = line.split('¶')[0].strip()
            entry_data = [application_number]
            for tag in tags_and_columns.keys():
                entry_data.append(extract_text(tag, line))
            entries.append(entry_data)

# Convert the list of entries to a DataFrame
column_names = ['출원번호'] + list(tags_and_columns.values())
df = pd.DataFrame(entries, columns=column_names)

# Read Abstract txt file
file_path = './DBII_000000000000003/TXT/Abstract.txt'
entries = []  # List to store each entry's data
df['요약'] = pd.NA

with open(file_path, 'r', encoding='utf-8') as file:
    for i, line in enumerate(file):
        if i> 0:
            application_number = line.split('¶')[0].strip()
            abstract = line.split('¶')[1].strip()
            abstract = delete_tag(abstract)
            df.loc[df['출원번호'] == application_number, '요약'] = abstract

# Read Bibliographic txt file
file_path = './DBII_000000000000003/TXT/Bibliographic.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    for i, line in enumerate(file):
        if i==0:
            column_list = line.split('¶')
        else:
            application_number = line.split('¶')[0].strip()
            for i, col in enumerate(column_list):
                content = line.split('¶')[i].strip()
                content = delete_tag(content)
                df.loc[df['출원번호'] == application_number, col] = content


# Read Claim txt file
file_path = './DBII_000000000000003/TXT/Claim.txt'
entries = []  # List to store each entry's data
df['청구항'] = pd.NA

with open(file_path, 'r', encoding='utf-8') as file:
    for i, line in enumerate(file):
        if i> 0:
            application_number = line.split('¶')[0].strip()
            content = line.split('¶')[1].strip()
            content = delete_tag(content)
            df.loc[df['출원번호'] == application_number, '청구항'] = content

# Read cpc txt file
file_path = './DBII_000000000000003/TXT/CPC.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    for i, line in enumerate(file):
        if i==0:
            column_list = line.split('¶')
        else:
            application_number = line.split('¶')[0].strip()
            for i, col in enumerate(column_list):
                content = line.split('¶')[i].strip()
                content = delete_tag(content)
                df.loc[df['출원번호'] == application_number, col] = content
# Read ipc txt file
file_path = './DBII_000000000000003/TXT/IPC.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    for i, line in enumerate(file):
        if i==0:
            column_list = line.split('¶')
        else:
            application_number = line.split('¶')[0].strip()
            for i, col in enumerate(column_list):
                content = line.split('¶')[i].strip()
                content = delete_tag(content)
                df.loc[df['출원번호'] == application_number, col] = content

# Read PriorTechnologyDocument.txt
file_path = './DBII_000000000000003/TXT/PriorTechnologyDocument.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    for i, line in enumerate(file):
        if i==0:
            column_list = line.split('¶')
        elif i == 1:
            application_number= line.split('¶')[0].strip()
            content = line.split('¶')[2].strip()
        else:
            if int(line.split('¶')[1].strip()) == 1:
                df.loc[df['출원번호'] == application_number, '선행연구']=content

                # Reset the number and gold list
                application_number= line.split('¶')[0].strip()
                content = line.split('¶')[2].strip()
            else:
                content+=' '+line.split('¶')[2].strip()


# Specify the output CSV file path
output_csv_path = './extracted_data_formatted.csv'
output_xlsx_path = './extracted_data_formatted.xlsx'

# Write to CSV
df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
df.to_excel(output_xlsx_path)