import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Difference in hours between timestamps
def time_diff(first,last):  
    delay = datetime.strptime(last,'%Y-%m-%d %H:%M:%S') - datetime.strptime(first,'%Y-%m-%d %H:%M:%S')
    return int((delay.days*24)+(delay.seconds/3600))/24


# Read in admissions table
df = pd.read_csv("/Volumes/Samir\'s Files/DATA/mimic/ADMISSIONS.csv.gz")

# Extract admit times and expiration times
times = df[['ADMITTIME','DEATHTIME']]
times = np.array(times.dropna())

# Calculate time to expiration from admission time
hours = []
for admit,expire in times:
    diff = time_diff(admit,expire)
    if diff > 0:
        hours.append(diff)
hours = np.array(hours)

# Plot distribution
plt.figure(1)
sns.distplot(hours)
plt.show()

# Plot histogram
plt.figure(2)
sns.distplot(hours, kde=False)
plt.show()

