from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from chatbot import Chatbot

app = FastAPI()

class Query(BaseModel):
    query: str
    chat_history: list[str] = []

class HealthCheck(BaseModel):
    status: str

@app.post("/generate")
async def generate_response(query: Query):
    try:
        response = app.state.chatbot.generate_response(query.query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthCheck)
async def health_check():
    return {"status": "OK"}

def serve_fastapi(chatbot: Chatbot, host: str, port: int):
    import uvicorn
    app.state.chatbot = chatbot
    uvicorn.run(app, host=host, port=port)