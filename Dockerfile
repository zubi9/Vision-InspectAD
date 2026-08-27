FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY api/ ./api/

# Model weights are NOT baked into the image -- they're large, change independently
# of code, and mounting them (see docker-compose.yml's ./models:/app/models volume)
# means a fresh clone with no trained weights yet still builds successfully, which
# also matters for CI (models/ is gitignored and won't exist on a checkout).
RUN mkdir -p models

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
