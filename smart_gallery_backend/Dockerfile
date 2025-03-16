# Use minimal Python image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Setup container directories
WORKDIR /app

# Copy local code to the container
COPY . /app

# Install libraries (use a virtual environment)
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir -r /app/requirements.txt

# Set PATH to use venv
ENV PATH="/opt/venv/bin:$PATH"

# Expose the port
EXPOSE 8080

# Launch the server with Gunicorn
CMD ["gunicorn", "main:app", "--timeout=0", "--preload", \
     "--workers=1", "--threads=4", "--bind=0.0.0.0:8080", "-k", "uvicorn.workers.UvicornWorker"]
