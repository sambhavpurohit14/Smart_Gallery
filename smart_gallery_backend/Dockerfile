FROM python:3.9-slim

# Install libraries
COPY ./requirements.txt ./
RUN pip install -r requirements.txt && \
    rm ./requirements.txt

# Setup container directories
RUN mkdir /app

# Copy local code to the container
COPY . /app

# Set the working directory
WORKDIR /app

# Expose the port
EXPOSE 8080

# Launch the server with gunicorn
CMD ["gunicorn", "main:app", "--timeout=0", "--preload", \
     "--workers=1", "--threads=4", "--bind=0.0.0.0:8080", "-k", "uvicorn.workers.UvicornWorker"]