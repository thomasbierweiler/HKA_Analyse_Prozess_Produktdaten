# read data from csv-file 'SmA-Four-Tank-Batch-Process_V2.csv' and determine the length of the SFC steps
# code follows the example given by Tobias Kühn

# Please copy the file 'SmA-Four-Tank-Batch-Process_V2.csv' to the current directory before executing the script

import os.path
import pandas as pd

fname='SmA-Four-Tank-Batch-Process_V2.csv'
#fname='StepNoSmall.csv'
if not os.path.isfile(fname):
    print('Missing file {}. See file Readme.txt for further information.'.format(fname))
    exit()
df=pd.read_csv(fname,sep=';')
df['timestamp']=pd.to_datetime(df['timestamp'],format='%Y-%m-%d %H:%M:%S')
if 'DeviationID ValueY' in df.columns:
    print('Number of deviations: {}'.format(len(df['DeviationID ValueY'].unique())))
print('Number of SFC steps: {}'.format(len(df['CuStepNo ValueY'].unique())))

# find start of SFC steps
df['step_s']=df['CuStepNo ValueY'].shift(1)
df['start']=df['CuStepNo ValueY']!=df['step_s']
dfStepStart=df.loc[(df['start']==True)]
# find end of SFC steps
df['step_e']=df['CuStepNo ValueY'].shift(-1)
df['end']=df['CuStepNo ValueY']!=df['step_e']
dfStepEnd=df.loc[(df['end']==True)]
# combine start and end dataframes
dff=dfStepStart.copy(deep=True)
dff['timestamp_last']=dfStepEnd['timestamp'].to_numpy()
dff['duration']=dff['timestamp_last']-dff['timestamp']
# drop some columns
dff.drop(['step_s','start'],axis=1,inplace=True)

# print information about first steps
print(dff[['timestamp','duration','CuStepNo ValueY']].head())

print('Duration of steps in s')
print('Step\tmin\tmax\tmean\tmedian')
for step in df['CuStepNo ValueY'].unique():
    dfc=dff[dff['CuStepNo ValueY']==step]
    print('{}\t{}\t{}\t{}\t{}'.format(step,dfc['duration'].min().total_seconds(),dfc['duration'].max().total_seconds(),\
        dfc['duration'].mean().total_seconds(),dfc['duration'].median().total_seconds()))