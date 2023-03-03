import re 
import pandas as pd 

def get_re_tanggal(x):
    try:
        return re.findall(r"\d{4}\/\d{2}\/\d{2}", x)[0]
    except:
        return None  

def get_csv_by_date(month, date):
    df = pd.read_csv(f'output_csv_master/FD_non_face_result_{month}_{date}.csv', names=['source','kode_voucher','tanggal_foto','KCU','KPM','hasil_prediksi'])
    df['hasil_prediksi'] = False
    return df