import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from gemma_rag_system import GemmaRAGSystem
from pydantic import BaseModel
from typing import Optional, List, Union

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Initialize the RAG system
rag_system = GemmaRAGSystem()

class QuestionRequest(BaseModel):
    question: str
    file_filter: Optional[Union[str, List[str]]] = None
    conversation_history: Optional[List[dict]] = None

@app.post("/generate-answer")
async def answer_question(request: QuestionRequest):
    """
    Answer a question using the RAG system
    
    Args:
        String question, optional file_filter, and optional conversation_history
        
    Returns:
        dict: Response containing answer, sources, and metadata
    """
    try:
        result = rag_system.answer_question(request.question,
                                            request.file_filter,
                                            request.conversation_history
                                            )
        return result
    except Exception as e:
        return {"error": str(e), "answer": "", "sources": []}

@app.get("/")
async def root():
    """Root endpoint to check if the API is running"""
    return {"message": "LeapLogic RAG API is running", "version": "1.0.0"}

@app.get("/get-statistics")
async def get_statistics():
    """Health check endpoint"""
    try:
        result = rag_system.get_statistics()
        return result
    except Exception as e:
        return {"error": str(e), "answer": "", "sources": []}

@app.get("/get-model-name")
async def get_model_name():
    """Endpoint to get the name of the model being used"""
    try:
        model_name = rag_system.get_model_name()
        return {"model_name": model_name}
    except Exception as e:
        return {"error": str(e), "model_name": ""}
    
@app.post("/reload-knowledge-base")
async def reload_knowledge_base():
    """Endpoint to reload the knowledge base"""
    try:
        rag_system.reload_knowledge_base()
        return {"message": "Knowledge base reloaded successfully"}
    except Exception as e:
        return {"error": str(e)}