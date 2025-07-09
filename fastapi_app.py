from fastapi import FastAPI, Request
from pydantic import BaseModel
from rag_agent import CarManualRAGAgent
from typing import Optional
import uvicorn

app = FastAPI()

# For demo: store a single agent instance and chat history
global_agent = None

def get_agent(api_key: str):
    global global_agent
    if global_agent is None:
        global_agent = CarManualRAGAgent(api_key=api_key, pdf_path="./car_manual.pdf")
    return global_agent

class QuestionRequest(BaseModel):
    question: str
    api_key: str

class AnswerResponse(BaseModel):
    answer: str

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(req: QuestionRequest):
    agent = get_agent(req.api_key)
    answer = agent.ask(req.question)
    return {"answer": answer}

if __name__ == "__main__":
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=8000, reload=True) 