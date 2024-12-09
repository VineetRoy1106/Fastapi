



# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# def process_pdf(file: io.BytesIO):
#     """
#     Processes the uploaded PDF, extracts text, and generates embeddings for each document.
#     """
#     try:
#         # Save the uploaded PDF to a temporary file
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
#             temp_file.write(file.read())
#             temp_file_path = temp_file.name

#         # Load the PDF using PyPDFLoader
#         pdf_loader = PyPDFLoader(temp_file_path)
#         docs = pdf_loader.load()

#         # Extract text content from each document
#         dataset = [doc.page_content for doc in docs]

#         # Generate embeddings for the dataset
#         embeddings_matrix = embeddings.embed_documents(dataset)

#         # Return both the raw dataset and its embeddings
#         return {"dataset": dataset, "embeddings": embeddings_matrix}

#     except Exception as e:
#         raise Exception(f"Error processing PDF: {str(e)}")


# def ask_question(query: str, dataset: list):
#     """
#     Accepts a query, computes cosine similarity with the documents in the dataset,
#     and retrieves the most relevant document(s).
#     """
#     try:
#         # Step 1: Get embedding for the query
#         query_embedding = embeddings.embed_documents([query])[0]  # Embed the query as a document

#         # Step 2: Generate embeddings for the dataset documents
#         document_embeddings = embeddings.embed_documents(dataset)  # Embed the dataset documents

#         # Step 3: Compute cosine similarity between the query and document embeddings
#         similarities = cosine_similarity([query_embedding], document_embeddings)[0]  # Flatten array

#         # Step 4: Find the indices of the most similar documents (you can return top N results)
#         top_indices = np.argsort(similarities)[::-1]  # Sort indices in descending order of similarity
#         most_similar_index = top_indices[0]  # Most similar document index

#         # Step 5: Retrieve the most similar document
#         most_similar_doc = dataset[most_similar_index]

#         # Optionally, aggregate content for top-N documents
#         top_n = 3
#         top_docs = [dataset[idx] for idx in top_indices[:top_n]]
#         combined_content = "\n\n".join(top_docs)

#         # Return the most similar document or combined content
#         return {
#             "most_similar_doc": most_similar_doc,
#             "combined_top_docs": combined_content,
#         }

#     except Exception as e:
#         raise Exception(f"Error answering question: {str(e)}")


import io
import tempfile
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader

# Initialize the embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")



def process_pdf(file: io.BytesIO):
    """
    Processes the uploaded PDF, extracts text, and generates embeddings for each document.
    """
    try:
        # Save the uploaded PDF to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file.read())
            temp_file_path = temp_file.name

        # Load the PDF using PyPDF2
        pdf_reader = PdfReader(temp_file_path)
        dataset = []
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                dataset.append(text)

        # Generate embeddings for the dataset
        embeddings_matrix = embeddings.embed_documents(dataset)

        # Return the raw dataset and its embeddings
        return {"dataset": dataset, "embeddings": embeddings_matrix}

    except Exception as e:
        raise Exception(f"Error processing PDF: {str(e)}")

# def process_pdf(file: io.BytesIO):
#     """
#     Processes the uploaded PDF, extracts text, and generates embeddings for each document.
#     """
#     try:
#         # Save the uploaded PDF to a temporary file
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
#             temp_file.write(file.read())
#             temp_file_path = temp_file.name

#         # Load the PDF using PyPDFLoader
#         pdf_loader = PyPDFLoader(temp_file_path)
#         docs = pdf_loader.load()

#         # Extract text content from each document
#         dataset = [doc.page_content for doc in docs]

#         # Generate embeddings for the dataset
#         embeddings_matrix = embeddings.embed_documents(dataset)

#         # Return the raw dataset and its embeddings
#         return {"dataset": dataset, "embeddings": embeddings_matrix}

#     except Exception as e:
#         raise Exception(f"Error processing PDF: {str(e)}")


def ask_question(query: str, dataset: list, embeddings_matrix: np.ndarray):
    """
    Accepts a query, computes cosine similarity with the documents in the dataset,
    and retrieves the most relevant document(s).
    """
    try:
        # Step 1: Get embedding for the query
        query_embedding = embeddings.embed_query(query)  # Use the correct embedding method for queries

        # Step 2: Compute cosine similarity between the query and document embeddings
        similarities = cosine_similarity([query_embedding], embeddings_matrix)[0]

        # Step 3: Find the indices of the most similar documents (e.g., top N results)
        top_indices = np.argsort(similarities)[::-1]  # Sort indices in descending order of similarity
        most_similar_index = top_indices[0]  # Most similar document index

        # Retrieve the most similar document
        most_similar_doc = dataset[most_similar_index]

        # Optionally, aggregate content for top-N documents
        top_n = 1
        top_docs = [dataset[idx] for idx in top_indices[:top_n]]
        combined_content = "\n\n".join(top_docs)

        # Return the most similar document or combined content
        return {
            "most_similar_doc": most_similar_doc,
            "combined_top_docs": combined_content,
        }

    except Exception as e:
        raise Exception(f"Error answering question: {str(e)}")
