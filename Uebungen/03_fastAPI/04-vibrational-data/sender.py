from fastapi import FastAPI
import json

app = FastAPI()

vib_data=((1,2,3),(4,5,6),(6,7,9))

# decorator with (URL) path definition (endpoint)
@app.get("/")
# asynchronous (non-blocking) call to method
async def root():
    return {"message": "Send vibrational data"}

# decorator with parameterized endpoint
@app.get("/sender")
async def sender():
    jsonString=json.dumps(vib_data)
    return jsonString
