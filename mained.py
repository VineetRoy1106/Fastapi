# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel
# import io
# from pdf_scraper import process_pdf, ask_question

# # FastAPI app setup
# app = FastAPI()

# class QueryRequest(BaseModel):
#     query: str

# @app.post("/process_pdf/")
# async def process_pdf_endpoint(file: UploadFile = File(...)):
#     try:
#         # Read the uploaded PDF file
#         file_content = await file.read()
#         pdf_file = io.BytesIO(file_content)

#         # Process the PDF using pdf_scraper
#         extracted_text = process_pdf(pdf_file)

#         return JSONResponse(content={"message": "PDF content processed successfully", "extracted_text": extracted_text})

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/ask_question/")
# async def ask_question_endpoint(query_request: QueryRequest):
#     try:
#         query = query_request.query
        
#         # Use pdf_scraper to process the query and get the answer
#         answer = ask_question(query)

#         return JSONResponse(content={"message": "Query processed successfully", "answer": answer})

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
import io
# from pdf_scraper import process_pdf, ask_question
from pdfscraper2 import process_pdf, ask_question
import uuid 

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

        chat_id = str(uuid.uuid4())

        return JSONResponse(content={ "chat_id": chat_id,"message": "PDF content processed and stored successfully", "extracted_text": extracted_text})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @app.post("/chat")
# async def ask_question_endpoint(query_request: QueryRequest):
#     try:
#         query = query_request.query
        
#         # Use pdf_scraper to process the query and get the answer
#         answer = ask_question(query)

#         return JSONResponse(content={"message": f"The main idea of the document is . . . {answer}"})
    
#         # return JSONResponse(content={"message": "Query processed successfully", "answer": answer})

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

from typing import Dict
pdf_storage: Dict[str, str] = {}

# Mock embedding function
def generate_embedding(text: str) -> np.ndarray:
    # Replace with an actual embedding generation function like OpenAI's embeddings or similar
    return np.random.rand(300)  # Mock 300-dimensional embedding

# Mock cosine similarity calculation
def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


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

        # Retrieve the stored content
        # document_text = pdf_storage[chat_id]

        # Generate embeddings for the document and the question
        answer = ask_question(question)

        # # Calculate cosine similarity (mocked logic)
        # similarity_score = cosine_similarity(document_embedding, question_embedding)

        

        return JSONResponse(content={"message": f"The main idea of the document is . . . {answer}"})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
