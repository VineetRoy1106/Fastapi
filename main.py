from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from webscraper import Scraper
from pydantic import BaseModel
from fastapi import FastAPI
import numpy as np
import uuid
import io
# from pdf_scraper import process_pdf, ask_question
from pdfscraper2 import process_pdf, ask_question


from typing import Dict
pdf_storage: Dict[str, str] = {}

app = FastAPI()

# Define the request body structure
class URLRequest(BaseModel):
    url: str

# Define the response structure
class ResponseModel(BaseModel):
    chat_id: str
    message: str

@app.post("/process_url", response_model=ResponseModel)
async def process_url(request: URLRequest):
    # Initialize the scraper
    scraper = Scraper()

    # Scrape data from the provided URL
    scraped_data = scraper.scrape_data(request.url)

    # If there is an error in scraping, return an error message
    if "error" in scraped_data:
        return {"chat_id": "", "message": scraped_data["error"]}

    # Create a unique chat_id
    chat_id = str(uuid.uuid4())

    # Here you can save the scraped content and associate it with the chat_id if needed.
    # For simplicity, we will just return the success message.

    return {"chat_id": chat_id, "message": "URL content processed and stored successfully."}


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

