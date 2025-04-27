FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install YOLOv8 via ultralytics
RUN pip install --no-cache-dir ultralytics

# Install Label Studio SDK
RUN pip install --no-cache-dir label-studio-sdk

# Download a pre-trained YOLOv8 model
RUN wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

# Create directory structure
RUN mkdir -p /app/shared-data/new_images \
    /app/shared-data/yolo_predictions \
    /app/shared-data/corrected_data \
    /app/shared-data/yolo_training

# Copy the application code
COPY . /app/

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Command to run the application
CMD ["python", "active_learning.py"]