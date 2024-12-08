FROM python:3.9-slim

# Set the working directory
WORKDIR /mained

# Copy the application code
COPY . /mained

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

# Run the application
CMD ["uvicorn", "mained:app", "--host", "0.0.0.0", "--port", "8000"]
