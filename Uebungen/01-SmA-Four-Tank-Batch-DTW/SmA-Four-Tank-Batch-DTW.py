# show the usage of dynamic time warping

# Please copy the file 'SmA-Four-Tank-Batch-Process_V2.csv' to the current directory before executing the script

import os.path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from dtaidistance import dtw

fname='SmA-Four-Tank-Batch-Process_V2.csv'
#fname='StepNoSmall.csv'
if not os.path.isfile(fname):
    print('Missing file {}.'.format(fname))
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

# apply DTW to measurement value for SFC step 1
step=1
dfc=dff[dff['CuStepNo ValueY']==step]
fig,ax=plt.subplots(2)
ax[0].set_title('original level')
ax[1].set_title('warped level (warped against level of first run)')
dfs_f=df.loc[(df['timestamp']>=dfc['timestamp'].iat[0]) & (df['timestamp']<=dfc['timestamp_last'].iat[0])]
f_run=dfs_f['LIC21002_PV_Out ValueY'].to_numpy()
ax[0].plot(dfs_f['timestamp'],dfs_f['LIC21002_PV_Out ValueY'])
ax[1].plot(dfs_f['timestamp'],dfs_f['LIC21002_PV_Out ValueY'])
for s in range(1,min([20,dfc.shape[0]])):
    # get data for first step
    dfs=df.loc[(df['timestamp']>=dfc['timestamp'].iat[s]) & (df['timestamp']<=dfc['timestamp_last'].iat[s])]
    # determine time offset
    offset=dfs['timestamp'].min()-df['timestamp'].min()
    # plt.plot(dfs['timestamp']-offset,dfs['CuStepNo ValueY'])
    ax[0].plot(dfs['timestamp']-offset,dfs['LIC21002_PV_Out ValueY'])
    # warp level with respect to level of first run
    level=dfs['LIC21002_PV_Out ValueY'].to_numpy()
    d, paths=dtw.warping_paths(f_run, level)
    best_path=dtw.best_path(paths)
    warped_path=np.zeros(shape=f_run.shape)
    # for best_path, warp level to level of first run 
    for el in best_path:
        warped_path[el[0]]=level[el[1]]
    ax[1].plot(dfs_f['timestamp'],warped_path)
plt.show()
print('Done')