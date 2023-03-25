from fastapi import FastAPI
from enum import Enum
import modules.read_data as rd
import modules.determine_step_length as dsl

app = FastAPI()

class batch_processing_state(Enum):
    not_processed=0
    reading=1
    reading_finished=2
    interpreting=3
    processed=4

# in-memory database; for a productive system, a real database should be used
db_df={'df' : None,'df_info_batches' : None,'df_info_steps' : None,'processing_state' : batch_processing_state.not_processed }

# decorator with (URL) path definition (endpoint)
@app.get("/")
# asynchronous (non-blocking) call to method
async def root():
    return {"message": "API for evaluation of timeseries data from 4-tank batch process."}

# endpoint for reading batch
@app.get("/read_batch")
async def read_batch():
    if db_df['processing_state']==batch_processing_state.not_processed:
        # data has not been read yet
        db_df['processing_state']=batch_processing_state.reading
        df=rd.read_batch_data()
        db_df['df']=df
        db_df['processing_state']=batch_processing_state.reading_finished
        return {"state": "reading of data scheduled"}
    else:
        return {"state": db_df['processing_state']}

# endpoint for processing data
@app.get("/process_data")
async def process_data():
    print("In process data")
    if db_df['processing_state']==batch_processing_state.reading_finished:
        # process data
        db_df['processing_state']=batch_processing_state.interpreting
        (df_info_batches,df_info_steps)=dsl.process_data(db_df['df'])
        db_df['df_info_batches']=df_info_batches
        db_df['df_info_steps']=df_info_steps
        db_df['processing_state']=batch_processing_state.processed
        return {"state": "interpreting timeseries data"}
    else:
        return {"state": db_df['processing_state']}

# endpoint for batch info
@app.get("/batch_info")
async def batch_info():
    if db_df['processing_state']==batch_processing_state.processed:
        dfib=db_df['df_info_batches']
        nbatches_complete=dfib['batchn'].max()
        nbatches_incomplete=dfib['batchn'].min()
        return {"state": db_df['processing_state'],
                "n_batches_complete" : int(nbatches_complete),
                "n_batches_incomplete" : int(nbatches_incomplete),
                "unit_batch_length" : "seconds",
                "longest_batch" : dfib['batch_length'].max().total_seconds(),
                "shortest_batch" : dfib['batch_length'].min().total_seconds()}
    else:
        return {"state": db_df['processing_state']}
    
# endpoint for step info
@app.get("/step_info/{step_n}")
async def step_info(step_n: int):
    if db_df['processing_state']==batch_processing_state.processed:
        tmp=db_df['df_info_steps'][db_df['df_info_steps']['stepn']==step_n]
        return {"state": db_df['processing_state'],
                "step_n": step_n,
                "unit_batch_length" : "seconds",
                "longest_step" : tmp['step_length'].max().total_seconds(),
                "shortest_step" : tmp['step_length'].min().total_seconds()}
    else:
        return {"state": db_df['processing_state']}
