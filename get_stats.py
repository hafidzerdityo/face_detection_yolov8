from os import listdir
from os.path import isfile, join
import json
import pandas as pd
from tqdm import tqdm

def get_face_stat():
    with open("face_stats.json", "r") as infile:
        stats = json.load(infile)
    print(stats)
        
    return stats