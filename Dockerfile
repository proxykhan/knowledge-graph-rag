# The API + web interface. Databases come from docker-compose.yml.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/app/.hf-cache
WORKDIR /app

# CPU-only PyTorch first: the default wheel bundles CUDA and is several GB larger.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY requirements.txt pyproject.toml ./
RUN pip install -r requirements.txt

COPY src ./src
COPY docs ./docs
COPY eval ./eval
RUN pip install --no-deps -e .

EXPOSE 8000
CMD ["uvicorn", "kgrag.api:app", "--host", "0.0.0.0", "--port", "8000"]
