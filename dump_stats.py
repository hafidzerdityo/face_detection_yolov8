import pandas as pd
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import json

import schedule
import time
import requests
import datetime


print(f'Stat job, started at {datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}!')
def job():
    mypath = 'input_csv_master/'
    list_dataset_bansos = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    list_dataset_bansos = [mypath + i for i in list_dataset_bansos]
    untested = 0
    tested = 0
    result_accepted = 0
    result_rejected = 0
    for i in tqdm(list_dataset_bansos):
        temp_df = pd.read_csv(i)
        untested += temp_df[temp_df['is_tested'] == False].shape[0]
        tested += temp_df[temp_df['is_tested'] == True].shape[0]
        result_accepted += temp_df[temp_df['result'] == True].shape[0]
        result_rejected += temp_df[temp_df['result'] == False].shape[0]

    stats_to_be_dumped = {
            'tested': tested,
            'untested': untested, 
            'result_accepted': result_accepted,
            'result_rejected': result_rejected}
    headers = {"Authorization": "Bearer " + "dGVuc29ycG9zX2hvdXNlX2RldGVjdGlvbg=="}

    house_stat_update = requests.get('http://10.25.11.175:8001/update_stats', headers=headers)
    house_stat_update = house_stat_update.json()
    print(f'face stats query-ed! {stats_to_be_dumped}')
    print(f'house stats query-ed! {house_stat_update}')
    # Serializing json
    json_object = json.dumps(stats_to_be_dumped, indent=4)
    
    # Writing to sample.json
    with open("face_stats.json", "w") as outfile:
        outfile.write(json_object)

# schedule.every(1).minutes.do(job)
schedule.every().hour.do(job)
# schedule.every().day.at("10:30").do(job)
# schedule.every().monday.do(job)
# schedule.every().wednesday.at("13:15").do(job)
# schedule.every().minute.at(":17").do(job)

while True:
    schedule.run_pending()
    time.sleep(1)
