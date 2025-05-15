# Dockerfile for Azure AI Mega Toolkit

# 1. Base Image
# Use an official Python runtime as a parent image.
# python:3.11-slim-bullseye is a good choice as 'bullseye' (Debian 11)
# has a newer GLIBC (2.31) which should resolve issues with azure-cognitiveservices-speech.
FROM python:3.11-slim-bullseye

# 2. Set Environment Variables
# These can be overridden by Azure App Service Application Settings
ENV PYTHONUNBUFFERED 1
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV PORT=8000
# Gunicorn settings (can also be set/overridden in App Service)
ENV GUNICORN_WORKERS=${GUNICORN_WORKERS:-2}
ENV GUNICORN_THREADS=${GUNICORN_THREADS:-2}
ENV GUNICORN_TIMEOUT=${GUNICORN_TIMEOUT:-300}
# NLTK data path within the container
ENV NLTK_DATA=/app/nltk_data_local

# 3. Create and Set Working Directory
WORKDIR /app

# 4. Install System Dependencies
# Update package lists and install essential build tools,
# libraries for Azure SDKs (like libssl for SSL, libffi),
# and libgomp1 which is a runtime dependency for azure-cognitiveservices-speech.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    curl \
    gnupg \
    openssl \
    ca-certificates \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 5. Copy and Install Python Requirements
# Copy requirements.txt first to leverage Docker layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy Application Code
# Copy the rest of your application code into the /app directory.
COPY . .

# 7. Create Application Directories and Set Permissions
# Create directories your application might need for uploads, instance data (SQLite), NLTK.
# Assign ownership to www-data if running Gunicorn as non-root (though default App Service runs as root in container).
# For simplicity, we'll rely on root execution initially.
RUN mkdir -p /app/instance /app/uploads /app/nltk_data_local && \
    chmod -R 755 /app/instance /app/uploads /app/nltk_data_local
    # If you were to run as a non-root user later:
    # chown -R www-data:www-data /app/instance /app/uploads /app/nltk_data_local

# 8. Expose Port
# Make port 8000 available to the world outside this container.
# Azure App Service will map its external port (80/443) to this container port.
EXPOSE 8000

# 9. Define the Command to Run the Application
# Use Gunicorn to run the Flask application (app:app means the 'app' object in 'app.py').
# Logs are directed to stdout/stderr for easy capture by Docker and App Service.
# The PORT environment variable is used by Gunicorn.
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers=2", "--threads=2", "--timeout=300", "--access-logfile", "-", "--error-logfile", "-", "app:app"]
