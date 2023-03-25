import pandas as pd
dfinfo_steps=pd.read_pickle('SmA-Four-Tank-Info-Steps.pkl')
dfinfo_batches=pd.read_pickle('SmA-Four-Tank-Info-Batches.pkl')
print('#complete batches: {}'.format(dfinfo_steps['batchn'].max()))
print('#incomplete batches: {}'.format(abs(dfinfo_steps['batchn'].min())))
print('step#\tshortest\tlongest')
for v in set(dfinfo_steps['stepn']):
    tmp=dfinfo_steps[dfinfo_steps['stepn']==v]
    print('{}\t{}\t\t{}'.format(v,tmp['step_length'].min().total_seconds(),tmp['step_length'].max().total_seconds()))
print('Longest batch: {}'.format(dfinfo_batches['batch_length'].max()))
print('Shortest batch: {}'.format(dfinfo_batches['batch_length'].min()))
