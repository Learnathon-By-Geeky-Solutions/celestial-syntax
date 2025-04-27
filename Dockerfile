FROM python:3.12-slim

# Install system dependencies for OpenCV, dlib, and Python build tools
RUN apt-get update && apt-get install -y \
    cmake \
    gcc \
    g++ \
    make \
    pkg-config \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libice6 \
    libxrender1 \
    libxext6 \
    libx11-6 \
    libxcb1 \
    libxi6 \
    libxt6 \
    libxfixes3 \
    libxau6 \
    libxdmcp6 \
    libuuid1 \
    libglu1-mesa \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port (change if your app uses a different port)
EXPOSE 8080

# Run the Flask app with Gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080"]
