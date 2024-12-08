# # Base image
# # FROM python:3.9-slim
# # FROM python:3.10-alpine

# FROM python:3.7-slim-buster

# # Set the working directory in the container
# WORKDIR /mained

# # Copy the requirements file into the container
# COPY requirements.txt .



# # Install only pip, setuptools, wheel first (helpful for installing other packages more efficiently)
# RUN pip install --upgrade pip setuptools wheel

# # Install dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy the entire application code into the container
# COPY . .

# # Expose the application port
# EXPOSE 8000

# # Run the application
# CMD ["uvicorn", "mained:app", "--host", "0.0.0.0", "--port", "8000"]



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
