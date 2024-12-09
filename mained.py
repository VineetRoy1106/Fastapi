from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import io
import uuid
from pdf_scraper3 import process_pdf, ask_question
from webscraper import Scraper

# Dictionary to store processed PDF data (dataset and embeddings)
pdf_storage = {}

app = FastAPI()


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

        # Process the PDF
        extracted_data = process_pdf(pdf_file)

        # Generate a unique chat_id
        chat_id = str(uuid.uuid4())

        # Store the extracted dataset and embeddings
        pdf_storage[chat_id] = extracted_data

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

        # Retrieve the stored content and embeddings
        document_data = pdf_storage[chat_id]
        dataset = document_data["dataset"]
        embeddings_matrix = document_data["embeddings"]

        # Use the extracted text and embeddings to answer the question
        answer = ask_question(question, dataset, embeddings_matrix)

        # return JSONResponse(content={
        #     "response": f"The main idea of the document is:{answer["most_similar_doc"]},
        #     # "top_docs": answer["combined_top_docs"],
            
        # })

        return JSONResponse(content={
    "response": f"The main idea of the document is: {answer['most_similar_doc']}",
    # "top_docs": answer["combined_top_docs"],
    })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


import os

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "mained:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),  # Render assigns PORT via an environment variable
        log_level="info"
    )
