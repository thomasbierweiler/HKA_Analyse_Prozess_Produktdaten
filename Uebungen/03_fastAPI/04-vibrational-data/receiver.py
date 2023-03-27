# try http://127.0.0.1:8000/receive/[2,3,4]
# to send a list to the receiver

from fastapi import FastAPI
import json

app = FastAPI()

# decorator with (URL) path definition (endpoint)
@app.get("/")
# asynchronous (non-blocking) call to method
async def root():
    return {"message": "Receive vibrational data"}

# decorator with parameterized endpoint
@app.get("/receive/{data}")
async def receive(data):
    print("Received data.")
    dc=json.loads(data)
    print('Data: {}'.format(dc))
    print('Type of data: {}'.format(type(dc)))
    return "Received data."
