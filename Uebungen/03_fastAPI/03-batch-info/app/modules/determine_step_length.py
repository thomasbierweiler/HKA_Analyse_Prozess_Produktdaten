import pandas as pd

def process_data(df:pd.DataFrame):
    # remove steps data are not mentioned in the documentation
    vsteps=[1,7,8,3]
    df=df[df['CuStepNo ValueY'].isin(vsteps)]
    # determine start and end of steps
    df['dstep_p']=df['CuStepNo ValueY'].diff()
    df['dstep_n']=df['CuStepNo ValueY'].diff(-1)
    # select rows with a step change
    dfsen=df[(df['dstep_n']!=0)]
    dfsep=df[(df['dstep_p']!=0)]
    dfse=pd.concat([dfsen,dfsep])
    dfse=dfse.sort_values(by=['timestamp'])
    # create new dataframe where we store extracted information
    dfinfo_steps=pd.DataFrame(columns=['step_length','start','end','stepn'])
    # iterative approach
    pstep=-1
    c=0
    for n in range(dfse.shape[0]):
        # get row
        r=dfse.iloc[n]
        if pstep==r['CuStepNo ValueY']:
            # determine step length
            stepl=r['timestamp']-dfse.iloc[n-1]['timestamp']
            # update dataframe
            dfinfo_steps.loc[c]=(stepl,dfse.iloc[n-1]['timestamp'],r['timestamp'],r['CuStepNo ValueY'])
            c=c+1
        else:
            pstep=r['CuStepNo ValueY']
    # now determine whether the batch is complete
    batchn=1
    batchi=-1
    dfinfo_steps["batchn"]=0
    dfinfo_steps["is_complete"]=False
    dfinfo_batches=pd.DataFrame(columns=['batch_length','start','end','steps','batchn','is_complete'])
    n=0
    b=0
    while True:
        if n+len(vsteps)>dfinfo_steps.shape[0]:
            # complete info at incomplete, last batch
            steps=[]
            for v in range(dfinfo_steps.shape[0]-n):
                dfinfo_steps.at[n+v,'batchn']=batchi
                dfinfo_steps.at[n+v,'is_complete']=False
                steps.append(dfinfo_steps.at[n+v,'stepn'])
            dfinfo_batches.loc[b]=[dfinfo_steps.at[n+v,'end']-dfinfo_steps.at[n,'start'],dfinfo_steps.at[n,'start'], \
                                dfinfo_steps.at[n+v,'end'],steps,batchi,False]
            b=b+1
            break
        # check if all steps of a batch are present and in correct order
        isCorrect=True
        for v in range(len(vsteps)):
            isCorrect=dfinfo_steps.loc[n+v,'stepn']==vsteps[v]
            if not isCorrect:
                break
        if isCorrect:
            steps=[]
            for v in range(len(vsteps)):
                dfinfo_steps.at[n+v,'batchn']=batchn
                dfinfo_steps.at[n+v,'is_complete']=True
                steps.append(dfinfo_steps.at[n+v,'stepn'])
            dfinfo_batches.loc[b]=[dfinfo_steps.at[n+v,'end']-dfinfo_steps.at[n,'start'],dfinfo_steps.at[n,'start'], \
                                dfinfo_steps.at[n+v,'end'],steps,batchn,True]
            n=n+len(vsteps)
            batchn=batchn+1
            b=b+1
        else:
            steps=[]
            for vc in range(v):
                dfinfo_steps.at[n+vc,'batchn']=batchi
                dfinfo_steps.at[n+vc,'is_complete']=False
                steps.append(dfinfo_steps.at[n+v,'stepn'])
            dfinfo_batches.loc[b]=[dfinfo_steps.at[n+vc,'end']-dfinfo_steps.at[n,'start'],dfinfo_steps.at[n,'start'], \
                                dfinfo_steps.at[n+vc,'end'],steps,batchi,False]
            batchi=batchi-1
            n=n+vc
            b=b+1
    # save dfinfo_steps to file
    dfinfo_steps.to_pickle('SmA-Four-Tank-Info-Steps.pkl')
    dfinfo_batches.to_pickle('SmA-Four-Tank-Info-Batches.pkl')
    return (dfinfo_batches,dfinfo_steps)
