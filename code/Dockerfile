# File: Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV MOUNTED_MODEL_DIRECTORY=/app/models

EXPOSE 50051 8000

CMD ["python", "main.py"]
