FROM python:3.10-slim

# Install system dependencies required for pyzbar
RUN apt-get update && apt-get install -y \
    libzbar0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# IMPORTANT: Install PyTorch CPU directly to completely prevent GPU memory bloat
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . .

# Start the FastAPI web application, restricted to 1 worker to stay under 512MB RAM
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000", "--workers", "1"]
