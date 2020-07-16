from datetime import datetime

# Difference in hours between timestamps
def time_diff(first,last):  
    delay = datetime.strptime(last,'%Y-%m-%d %H:%M:%S') - datetime.strptime(first,'%Y-%m-%d %H:%M:%S')
    return int((delay.days*24)+(delay.seconds/3600))/24

