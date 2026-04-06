FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Exposed for Hugging Face Spaces & OpenEnv automated Docker validation check
EXPOSE 7860

# Boot the FastAPI via Uvicorn explicitly decoupled from baseline runner
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
