FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/       ./src/
COPY models/    ./models/
COPY reports/   ./reports/
COPY templates/ ./templates/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.predict:app", "--host", "0.0.0.0", "--port", "8000"]