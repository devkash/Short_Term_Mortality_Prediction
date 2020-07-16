from add_to_table import append
import pandas as pd

# Aggregating the SOFA, SAPISII, OASIS, APSIII and SAPS scores 
# into a single dataframe for easier manipulation

sofa = pd.read_csv('sofa_new.csv',delimiter='\t')
sofa = sofa.drop(columns=['subject_id'])
sofa = sofa.add_suffix('_sofa')

sapsii = pd.read_csv('sapsii_new.csv',delimiter='\t')
sapsii = sapsii.drop(columns=['subject_id'])
sapsii = sapsii.add_suffix('_sapsii')

oasis = pd.read_csv('oasis_new.csv',delimiter='\t')
oasis = oasis.drop(columns=['subject_id'])
oasis = oasis.add_suffix('_oasis')

apsiii = pd.read_csv('apsiii_new.csv',delimiter='\t')
apsiii = apsiii.drop(columns=['subject_id'])
apsiii = apsiii.add_suffix('_apsiii')

saps = pd.read_csv('saps_new.csv',delimiter='\t')
saps = saps.drop(columns=['subject_id'])
saps = saps.add_suffix('_saps')




all_scores = pd.concat([sofa,sapsii,oasis,apsiii,saps],axis=1)

append(all_scores,'all.csv')