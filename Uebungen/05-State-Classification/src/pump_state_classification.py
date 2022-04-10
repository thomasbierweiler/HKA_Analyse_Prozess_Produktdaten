# ROCKET: Dempster A, Petitjean F, Webb GI (2019) ROCKET: Exceptionally fast and accurate time series classification using random convolutional kernels. arXiv:1910.13051
#   https://arxiv.org/abs/1910.13051
# Demo of ROCKET transform: https://github.com/alan-turing-institute/sktime/blob/main/examples/rocket.ipynb

# !pip install --upgrade numba
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import RidgeClassifierCV

from sktime.transformations.panel.rocket import Rocket

# the multisensor measure structure-borne sound in three axis, x (B080), y (B081) and z (B082)

# features to use
features=['B080','B082']

# read data
df=None
for f in features:
    dfi=pd.read_csv('../data/2022-03-31/{}bin.csv'.format(f))
    if df is None:
        df=dfi
    else:
        # combine data from different measurement axis
        df[f]=dfi[f]

# convert ms since epoch in datetime
df['time']=pd.to_datetime(df['UnixEpochMilliseconds'],unit='ms')

# read log file
dfl=pd.read_csv('../data/PL23_log.csv')
dfl['start']=pd.to_datetime(dfl['Start'])
dfl['end']=pd.to_datetime(dfl['Ende'])
# subtract two hours --> UTC
dfl['start']=dfl['start']+pd.Timedelta(hours=-2)
dfl['end']=dfl['end']+pd.Timedelta(hours=-2)

# create labels from log file
df['label']='unspecified'
for i,s in dfl.iterrows():
    indx=(df['time']>=s['start']) & (df['time']<=s['end'])
    df.loc[indx,'label']=s['Zustand']

# remove entries with label unspecified
df=df[df['label']!='unspecified']

# use data till 17:00 (15:00 UTC) as training data
dfTr=df[df['time']<datetime(2022,3,31,15,0,0)]
# use data after 17:00 (15:00 UTC) as test data
dfTe=df[df['time']>=datetime(2022,3,31,15,0,0)]

# put data into a dataframe as expected by ROCKET
def transform_to_rocket_1feature(df:pd.DataFrame,id):
    df_Rocket=pd.DataFrame(columns=[id])
    df_Y=pd.DataFrame(columns=[id])
    df_Time=pd.DataFrame(columns=['time'])
    # create series for B000
    for g in df.groupby(by=['time']):
        indxC=df_Rocket.index.max()
        if indxC!=indxC:
            indxC=0
        else:
            indxC=indxC+1
        df_Rocket.loc[indxC]=[g[1][id]]
        df_Y=df_Y.append({id: g[1]['label'].iloc[0]}, ignore_index=True)
        df_Time=df_Time.append({'time': g[1]['time'].iloc[0]}, ignore_index=True)
    return df_Rocket,df_Y,df_Time

def transform_to_rocket(df,features):
    dfX=None
    dfY=None
    dfTime=None
    for id in features:
        dfX_,dfY_,dfTime_=transform_to_rocket_1feature(df,id)
        if dfX is None:
            dfX=dfX_
            dfY=dfY_
            dfTime=dfTime_
        else:
            dfX[id]=dfX_[id]
            # dfY[id]=dfY_[id] # assuming that for structure bourne sound all three axis are available
    return dfX, dfY, dfTime

# transform training data to ROCKET df
X_train,dfY_train,dfTime_train=transform_to_rocket(dfTr,features)
y_train=dfY_train.to_numpy()

# Initialise ROCKET and Transform the Training Data
rocket=Rocket()  # by default, ROCKET uses 10,000 kernels
rocket.fit(X_train)
X_train_transform=rocket.transform(X_train)

# Fit a Classifier
classifier=RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
classifier.fit(X_train_transform, y_train)

# Classify the Training Data
score_train=classifier.score(X_train_transform, y_train)
predictions_train=classifier.predict(X_train_transform)

# transform test data to ROCKET df
X_test,dfY_test,dfTime_test=transform_to_rocket(dfTe,features)
y_test=dfY_test.to_numpy()

# Transform the Test Data
X_test_transform=rocket.transform(X_test)

# Classify the Test Data
score_test=classifier.score(X_test_transform, y_test)
predictions_test=classifier.predict(X_test_transform)

print('Score for training data, features {}: {}'.format(features, score_train))
print('Score for test data, features {}: {}'.format(features, score_test))

print('Done')
