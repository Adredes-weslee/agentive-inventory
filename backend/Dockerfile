FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy dependency file and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY app/ ./app/

# Expose the port (matching the API_PORT environment variable)
ENV PYTHONPATH=/app
EXPOSE 8000

# Run the FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]