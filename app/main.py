from dotenv import load_dotenv
load_dotenv(override=True)



from fastapi import FastAPI

from app.api.ingestion import router as ingestion_router
from app.api.chat import router as chat_router

app = FastAPI()

app.include_router(ingestion_router)
app.include_router(chat_router)
