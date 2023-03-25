import pandas as pd

def read_batch_data()->pd.DataFrame():
    # import data to data frame
    df=pd.read_csv('SmA-Four-Tank-Batch-Process_V2.csv',delimiter=';')
    # timestamp is an object, not a timestamp
    df['timestamp']=pd.to_datetime(df['timestamp'],format='%Y-%m-%dT%H:%M:%S.%f')
    return df
