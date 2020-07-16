import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from time_diff import time_diff

filename = '/Volumes/Samir\'s Files/DATA/mimic/CHARTEVENTS.csv.gz'

new_hash = {}
test = [225698,220045]

count = 0
# USE LARGER CHUNKSIZE!
chunksize = 10
for chunk in pd.read_csv(filename, chunksize=chunksize):
    count += 1
    print(chunk)

    for index,row in chunk.iterrows():
        if row['ITEMID'] in test:
            new_hash[row['HADM_ID']] = row['VALUE']

    if count > 10:
        print(new_hash)
        quit()

