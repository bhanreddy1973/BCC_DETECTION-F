# Use official Ubuntu as a parent image
FROM ubuntu:22.04

# Set non-interactive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    build-essential \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy only the code and requirements, not data
COPY . /app

# Remove data directories if present (safety)
RUN rm -rf /app/data /app/patches /app/features || true

# Install Python dependencies
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# (Optional) Set environment variables
# ENV PYTHONPATH=/app

# Default command
CMD ["bash"] 