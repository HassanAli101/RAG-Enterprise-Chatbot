from fastapi import FastAPI, UploadFile
from EmployeeCentered import EmployeeChatBot
from pydantic import BaseModel

app = FastAPI()
bot = EmployeeChatBot()

class QueryRequest(BaseModel):
    query: str

class VerboseRequest(BaseModel):
    verbose: bool  # We expect a boolean value for the verbose flag


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

@app.post("/employee/changeVerbose")
def change_verbose(request: VerboseRequest):
    bot.verbose = request.verbose  # Set the verbose flag to the value sent in the request
    return {"verbose": bot.verbose}

@app.get("/employee/clearCache")
def clear_cache():
    bot.cache = []  # Clears the bot's cache
    return {"message": "Cache cleared successfully"}
