FROM python:3.10-slim

WORKDIR /app

# Install only what the inference API needs
COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

COPY app.py .

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
