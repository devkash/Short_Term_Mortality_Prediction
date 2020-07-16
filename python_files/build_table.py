import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from time_diff import time_diff

# Writing data to CSV file

df = pd.read_csv('/Volumes/Samir\'s Files/DATA/mimic/ADMISSIONS.csv.gz')
df['TARGET'] = 0
data = df[['HADM_ID','SUBJECT_ID','ADMITTIME','DEATHTIME','TARGET']]
data = data.replace(np.nan,'nan',regex=True)
col = data.columns.values.tolist()

data = np.array(data)

for i in range(len(data)):
    if data[i][3]!='nan':
        if time_diff(data[i][3],data[i][2]) <= 168:
            data[i][4] = 1


with open('test.csv', 'w') as csv_file:
    csvwriter = csv.writer(csv_file, delimiter='\t')
    csvwriter.writerow(col)
    for row in range(len(data)):
        csvwriter.writerow(data[row])