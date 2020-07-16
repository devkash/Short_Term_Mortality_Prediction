import pandas as pd
import numpy as np
import csv
from time_diff import time_diff

filename = 'all.csv'
new_filename = 'newfile.csv'
admittime = 2
deathtime = 3
target = 4

# FILE TO FIX
all_scores = pd.read_csv(filename,delimiter='\t')
col = all_scores.columns.values.tolist()

all_scores = np.array(all_scores)

corrected = 0
for i in range(len(all_scores)):
    adtime = all_scores[i][admittime]
    if all_scores[i][target] == 1:
        if time_diff(all_scores[i][admittime],all_scores[i][deathtime]) > 7:
            all_scores[i][target] = 0
            corrected += 1
print(corrected)

with open(new_filename, 'w') as csv_file:
    csvwriter = csv.writer(csv_file, delimiter='\t')
    csvwriter.writerow(col)
    for row in range(len(all_scores)):
        csvwriter.writerow(all_scores[row])
