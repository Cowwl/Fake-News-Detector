# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory to /app
WORKDIR /app

# Install dependencies first
COPY requirements.txt .

RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Install libgomp1 for the FastAPI app to run
RUN apt-get update
RUN apt-get upgrade
RUN apt-get install libgomp1

# Expose port 8000 for the FastAPI app to listen on
EXPOSE 8000

# Define the command to run the app when the container starts
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]