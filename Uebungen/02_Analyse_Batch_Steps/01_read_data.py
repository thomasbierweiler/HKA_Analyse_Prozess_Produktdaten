import pandas as pd
import datetime
import matplotlib.pyplot as plt

# import data to data frame
n=datetime.datetime.now()
df=pd.read_csv('SmA-Four-Tank-Batch-Process_V2.csv',delimiter=';')
n2=datetime.datetime.now()

# print first lines of data frame
print(df.head())
# print data types
print(df.dtypes)
# timestamp is an object, not a timestamp
df['timestamp']=pd.to_datetime(df['timestamp'],format='%Y-%m-%dT%H:%M:%S.%f')

print('Nach Umwandlung in ein Datumsformat:')
print('Erster Zeitstempel: {}'.format(df.at[1,'timestamp']))
print('Differenz der Zeitstempel: {}'.format(df['timestamp'].diff()))
print('Format des Zeitstempels: {}'.format(df['timestamp'].dtype))
# save data frame as pickle - more efficient
df.to_pickle('SmA-Four-Tank-Batch-Process_V2.pkl')
# read data frame from pickle
n3=datetime.datetime.now()
df=pd.read_pickle('SmA-Four-Tank-Batch-Process_V2.pkl')
n4=datetime.datetime.now()

print('Time to read data into data frame from csv: {}'.format(n2-n))
print('Time to read data into data frame from pickle: {}'.format(n4-n3))
# print columns of data frame
print(df.columns)
# plot column FIC14002_PV_Out ValueY (flow rate)
plt.subplot(121)
plt.plot(df['timestamp'],df['FIC14002_PV_Out ValueY'])
plt.ylabel('mass flow of F14002 / (kg/h)')
plt.xlabel('date')
# plot step number
plt.subplot(122)
plt.plot(df['timestamp'],df['CuStepNo ValueY'])
plt.ylabel('step number')
plt.xlabel('date')
plt.show()
# filter dataframe by step 1
df1=df[df['CuStepNo ValueY']==1]
print(df.describe())
print(df1.describe())
print('Shape of df (all steps): {}'.format(df.shape))
print('Shape of df1 (step 1): {}'.format(df1.shape))





