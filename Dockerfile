FROM python:3.11-slim AS builder

LABEL maintainer="Richard Gidi"
LABEL created="2025-07-30"

# Set the working directory
WORKDIR /app    

ENV PYTHONUNBUFFERED=1\
    PYTHONDONTWRITEBYTECODE=1\
    PIP_NO_CACHE_DIR=1

# Copy the requirements file
COPY requirements.txt . 

# Install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

ENV PYTHONUNBUFFERED=1\
    PYTHONDONTWRITEBYTECODE=1\
    PIP_NO_CACHE_DIR=1

# Copy the installed packages from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the application code
COPY . .

# Create models directory if it doesn't exist
RUN mkdir -p /app/models

EXPOSE 8000

# Command to run the application
CMD ["python", "-m", "gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "4", "app.main:app", "--bind", "0.0.0.0:8000"]