from fastapi import FastAPI, UploadFile
from EmployeeCentered import EmployeeChatBot
from pydantic import BaseModel

app = FastAPI()
bot = EmployeeChatBot()

class QueryRequest(BaseModel):
    query: str

@app.post("/employee")
async def employee_query(request: QueryRequest):
    print("the query is: ", request.query)  # Access the query from the request object
    response = bot.generate(request.query)
    return {"response": response}

@app.post("/employee/upload")
async def upload_document(file: UploadFile):
    temp_file_path = f"/tmp/{file.filename}"
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(await file.read())
    bot.AddFileToDB([temp_file_path])
    return {"message": f"File {file.filename} uploaded successfully"}
