# # pdf_scraper.py

# import io
# import nltk
# from langchain_huggingface import HuggingFaceEmbeddings
# # from langchain.embeddings import HuggingFaceEmbeddings
# from langchain_community.retrievers import PineconeHybridSearchRetriever
# from pinecone_text.sparse import BM25Encoder
# from langchain.chains.question_answering import load_qa_chain
# from langchain_groq import ChatGroq
# from langchain_community.document_loaders import PyPDFLoader
# from pinecone import Pinecone, ServerlessSpec
# import os
# from dotenv import load_dotenv

# # Download nltk data
# nltk.download('punkt')
# nltk.download('punkt_tab')

# # Load environment variables from .env file
# load_dotenv()

# # Load API keys from environment
# api_key = os.getenv("PINECONE_API_KEY")
# groq_api_key = os.getenv("GROQ_API_KEY")

# # Initialize Pinecone client and create index if it doesn't exist
# index_name = "pdfprocess"
# pc = Pinecone(api_key=api_key)

# # if index_name not in pc.list_indexes().names():
# #     pc.create_index(
# #         name=index_name,
# #         dimension=384,
# #         metric="dotproduct",
# #         spec=ServerlessSpec(cloud="azure", region="us-east-1")
# #     )
# index = pc.Index(index_name)

# # Embeddings and sparse matrix (BM25)
# embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
# bm25_encoder = BM25Encoder().default()

# import tempfile
# from langchain_community.document_loaders import PyPDFLoader

# def process_pdf(file: io.BytesIO):
#     """
#     Processes the uploaded PDF, extracts text, and indexes it in Pinecone.
#     """
#     try:
#         # Save the uploaded PDF to a temporary file
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
#             temp_file.write(file.read())
#             temp_file_path = temp_file.name
        
#         # Load the PDF using PyPDFLoader
#         pdf_loader = PyPDFLoader(temp_file_path)
#         docs = pdf_loader.load()
        
#         dataset = []
#         for doc in docs:
#             dataset.append(doc.page_content)

#         # Fit BM25 on the dataset
#         bm25_encoder.fit(dataset)

#         # Initialize hybrid search retriever
#         retriever = PineconeHybridSearchRetriever(
#             embeddings=embeddings, sparse_encoder=bm25_encoder, index=index
#         )
#         retriever.add_texts(dataset)

#         return dataset

#     except Exception as e:
#         raise Exception(f"Error processing PDF: {str(e)}")



# def ask_question(query: str):
#     """
#     Accepts a query, retrieves relevant documents from Pinecone, and generates an answer using a Groq-based model.
#     """
#     try:
#         # Retrieve similar documents using PineconeHybridSearchRetriever
#         retriever = PineconeHybridSearchRetriever(
#             embeddings=embeddings, sparse_encoder=bm25_encoder, index=index
#         )
#         similar_docs = retriever.invoke(query)

#         # Initialize Groq-based QA model
#         llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
#         chain = load_qa_chain(llm, chain_type="stuff")

#         # Retrieve the answer
#         answer = chain.invoke({"input_documents": similar_docs, "question": query})

#         return answer['output_text']

#     except Exception as e:
#         raise Exception(f"Error answering question: {str(e)}")



# pdf_scraper.py

# pdf_scraper.py

import io
import nltk
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone_text.sparse import BM25Encoder
from langchain.chains.question_answering import load_qa_chain
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv

# Download nltk data
nltk.download('punkt')
nltk.download('punkt_tab')

# Load environment variables from .env file
load_dotenv()

# Load API keys from environment
api_key = os.getenv("PINECONE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize Pinecone client and create index if it doesn't exist
index_name = "pdfprocess3"
pc = Pinecone(api_key=api_key)

# Check if the index already exists, otherwise create it with the correct dimensions
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # Set dimension to 1024 to match the new embedding size
        metric="dotproduct",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(index_name)

# Use a model that produces 1024-dimensional vectors
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # Updated model with 1024-dim
bm25_encoder = BM25Encoder().default()

import tempfile

def process_pdf(file: io.BytesIO):
    """
    Processes the uploaded PDF, extracts text, and indexes it in Pinecone.
    """
    try:
        # Save the uploaded PDF to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file.read())
            temp_file_path = temp_file.name
        
        # Load the PDF using PyPDFLoader
        pdf_loader = PyPDFLoader(temp_file_path)
        docs = pdf_loader.load()
        
        dataset = []
        for doc in docs:
            dataset.append(doc.page_content)

        # Fit BM25 on the dataset
        bm25_encoder.fit(dataset)

        # Initialize hybrid search retriever
        retriever = PineconeHybridSearchRetriever(
            embeddings=embeddings, sparse_encoder=bm25_encoder, index=index
        )
        retriever.add_texts(dataset)

        return dataset

    except Exception as e:
        raise Exception(f"Error processing PDF: {str(e)}")

def ask_question(query: str):
    """
    Accepts a query, retrieves relevant documents from Pinecone, and generates an answer using a Groq-based model.
    """
    try:
        # Retrieve similar documents using PineconeHybridSearchRetriever
        retriever = PineconeHybridSearchRetriever(
            embeddings=embeddings, sparse_encoder=bm25_encoder, index=index
        )
        similar_docs = retriever.invoke(query)

        # Initialize Groq-based QA model
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
        chain = load_qa_chain(llm, chain_type="stuff")

        # Retrieve the answer
        answer = chain.invoke({"input_documents": similar_docs, "question": query})

        return answer['output_text']

    except Exception as e:
        raise Exception(f"Error answering question: {str(e)}")


# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_groq import ChatGroq
# from langchain.chains.question_answering import load_qa_chain

# def ask_question(query: str, document_embeddings: np.ndarray):
#     """
#     Accepts a query, computes cosine similarity with the documents, retrieves the most relevant documents,
#     and generates an answer using a Groq-based model.
#     """
#     try:
#         # Get embedding for the query
#         query_embedding = embeddings.embed([query])

#         # Compute cosine similarity between the query and document embeddings
#         similarities = cosine_similarity(query_embedding, document_embeddings)

#         # Get the index of the most similar document (you can modify this to return top N results)
#         most_similar_index = np.argmax(similarities)

#         # Retrieve the most similar document
#         similar_doc = document_embeddings[most_similar_index]

#         # Initialize Groq-based QA model
#         llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
#         chain = load_qa_chain(llm, chain_type="stuff")

#         # Retrieve the answer based on the most similar document
#         answer = chain.invoke({"input_documents": [similar_doc], "question": query})

#         return answer['output_text']

#     except Exception as e:
#         raise Exception(f"Error answering question: {str(e)}")

