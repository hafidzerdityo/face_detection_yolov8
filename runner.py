import requests
from tqdm import tqdm
from datetime import datetime
import csv
import pandas as pd 
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import model.face_det_yv8 as face_det_yv8

url = 'http://127.0.0.1:8002/face_det_bansos'

def save_csv(src, file_name):
    csv_out_name = f'output_csv/FD_non_face_result_{datetime.now().strftime("%m")}_{datetime.now().strftime("%d")}.csv'
    data = [src,file_name]
    try:
        with open(csv_out_name, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(data)
    except FileNotFoundError as e:
        print(e)

def main(mypath):
    all_input_csv = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    all_input_csv = [mypath + i for i in all_input_csv]
    file_counter = 1
    for each_input_csv in all_input_csv:
        df_checker = pd.read_csv(each_input_csv)
        df_checker_false = df_checker[df_checker['is_tested'] == False]
        print(f'file: {file_counter}/{len(all_input_csv)}  nama file: {each_input_csv}')
        file_counter += 1
        try:
            for idx , val in tqdm(df_checker_false.iterrows(), total=df_checker_false.shape[0]):
                payload = {
                    "path" : val['src']
                }
                # resp = requests.post(url, json=payload)
                # data = resp.json()['data']
                data = face_det_yv8.predict_face_bansos(payload['path'])
                if not data['result']:
                    save_csv(val['src'], val['file_name']) 
                df_checker.loc[idx, 'is_tested'] = True
                df_checker.loc[idx, 'result'] = data['result']
                df_checker.to_csv(each_input_csv, index=False)
        except Exception as e:
            print('error:',e)
        finally:
            df_checker.to_csv(each_input_csv, index=False)

mypath = 'input_csv/'
main(mypath)