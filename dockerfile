# Base image
FROM python:3.9-slim

# Set environment variables for CI/CD (replace with actual keys in GitHub Actions or Azure)


# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application code into the container
COPY . .


ENV PINECONE_API_KEY=${PINECONE_API_KEY}
ENV GROQ_API_KEY=${GROQ_API_KEY}


RUN python -m nltk.downloader punkt
# Expose the application port
EXPOSE 8000



# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
