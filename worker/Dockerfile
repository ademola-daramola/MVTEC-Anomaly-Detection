# Build context is the project root (see docker-compose.yml)
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch keeps the image small. Swap the index URL for a CUDA build if you have a GPU host.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY worker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the ImageNet weights into the image so the container does not need internet at start-up
RUN python -c "from torchvision import models; models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)"

COPY common.py scorer.py ./
COPY worker/app.py ./

ENV MODELS_DIR=/models STORAGE_DIR=/storage

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
