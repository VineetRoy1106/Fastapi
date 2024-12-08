from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
import io
# from pdf_scraper import process_pdf, ask_question
from pdfscraper2 import process_pdf, ask_question
import uuid 

from typing import Dict
pdf_storage: Dict[str, str] = {}

# FastAPI app setup
app = FastAPI()

class QueryRequest(BaseModel):
    query: str


@app.post("/process_pdf")
async def process_pdf_endpoint(file: UploadFile = File(...)):
    try:
        # Read the uploaded PDF file
        file_content = await file.read()
        pdf_file = io.BytesIO(file_content)

        # Process the PDF using pdf_scraper
        extracted_text = process_pdf(pdf_file)

        # Generate a unique chat_id
        chat_id = str(uuid.uuid4())

        # Store the extracted text in pdf_storage
        pdf_storage[chat_id] = extracted_text

        return JSONResponse(content={
            "chat_id": chat_id,
            "message": "PDF content processed and stored successfully",
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
   
    

class ChatRequest(BaseModel):
    chat_id: str
    question: str


@app.post("/chat")
async def ask_question_endpoint(chat_request: ChatRequest):
    try:
        chat_id = chat_request.chat_id
        question = chat_request.question

        # Check if the chat_id exists in storage
        if chat_id not in pdf_storage:
            raise HTTPException(status_code=404, detail="Chat ID not found")

        # Retrieve the stored content using chat_id
        document_text = pdf_storage[chat_id]

        # Use the extracted text to answer the question
        answer = ask_question(question)

        return JSONResponse(content={"response": f"The main idea of the document is: {answer}"})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

