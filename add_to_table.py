import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time_diff import time_diff
import csv

# PASS IN:
#   pandas dataframe with HADM_ID as first column

def append(new_data,filename):
    new_col = new_data.columns.values.tolist()
    new_col.remove('HADM_ID')

    new_data = new_data.set_index('HADM_ID').T.to_dict('list')
    dummy = ['nan' for c in new_col]

    data = pd.read_csv('data.csv', delimiter='\t')
    col = data.columns.values.tolist()
    col.extend(new_col)
    data = data.set_index('HADM_ID').T.to_dict('list')

    for key, value in data.items():
        if key in new_data:
            data[key].extend(new_data[key])
        else:
            data[key].extend(dummy)


    with open(filename, 'w') as csv_file:
        csvwriter = csv.writer(csv_file, delimiter='\t')
        csvwriter.writerow(col)
        for key, value in data.items():
            row = [key]
            row.extend(value)
            csvwriter.writerow(row)