import uvicorn
from dotenv import load_dotenv

load_dotenv()

# import after loading env
from service import app

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)