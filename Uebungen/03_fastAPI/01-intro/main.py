from fastapi import FastAPI

app = FastAPI()

items_db=[{"student_name":"Julia"},{"student_name":"Max"},{"student_name":"Alex"}]

# decorator with (URL) path definition (endpoint)
@app.get("/")
# asynchronous (non-blocking) call to method
async def root():
    return {"message": "Hello World"}

# decorator with parameterized endpoint
@app.get("/students_name/{student_name}")
async def students_name(student_name):
    return {"student_name": "Student's name is " + student_name + "."}


# decorator with parameterized endpoint and type checking
@app.get("/nstudents/{n_of_students}")
async def students_name(n_of_students: int):
    return {"n_of_students": "#students: " + str(n_of_students) + "."}

# decorator with query paramters
@app.get("/read_students/")
async def read_students(skip: int=0, limit: int=10):
    return items_db[skip:skip+limit]