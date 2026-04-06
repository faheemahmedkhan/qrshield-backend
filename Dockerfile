FROM python:3.10-slim

# Install system dependencies required for pyzbar
RUN apt-get update && apt-get install -y \
    libzbar0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . .

# Start the FastAPI web application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
