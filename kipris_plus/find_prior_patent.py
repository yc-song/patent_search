import pandas as pd
import io


def main():
    file_path = '../kipris_process/prior_arts_formatted.csv'
    out_file_path = '../kipris_process/prior_arts_formatted.csv'

    df = pd.read_csv(file_path)

    # 일련번호 컬럼이 비어있고 특허유형 컬럼 형식이 잘못된 경우 수정
    for index, row in df.iterrows():
        if pd.isna(row['일련번호']) and pd.notna(row['특허유형']):
            patent_type_parts = row['특허유형'].split(' ')
            if len(patent_type_parts) == 2:
                df.at[index, '일련번호'] = patent_type_parts[0]
                df.at[index, '특허유형'] = patent_type_parts[1]
    for index, row in df.iterrows():
        if pd.notna(row['일련번호']) and row['일련번호'][-1].isalpha():
            # 알파벳을 분리하여 특허유형 앞에 추가
            alpha_part = row['일련번호'][-1]
            df.at[index, '일련번호'] = row['일련번호'][:-1]
            df.at[index, '특허유형'] = alpha_part + str(row['특허유형'])
    df['일련번호'] = df['일련번호'].str.replace('-', '')
    df['일련번호'] = df['일련번호'].apply(lambda x: f'{x}0000' if x.isdigit() and len(x) == 9 else x)
    # 수정된 DataFrame을 CSV 파일로 저장
    df.to_csv('prior_arts_formatted.csv', index=False)


if __name__ == '__main__':
    main()