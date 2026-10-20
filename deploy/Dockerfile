FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY configs/ ./configs/
COPY models/latest_model.pkl ./models/

EXPOSE 8000

CMD ["python", "deploy/service.py"]
