FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.hf \
    ROUTER_DIR=/app/arm

WORKDIR /app

# The pytorch cpu index declares no nvidia-* dependencies; PyPI's linux torch pulls ~3 GB of CUDA.
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.6,<3"

COPY requirements-serve.txt ./
RUN pip install -r requirements-serve.txt

# --no-deps: the wheel declares the whole research set, which requirements-serve.txt narrows.
# It also declares the query-taxonomy submodule, which the build never reads and serving
# never imports, so the submodule stays out of the image entirely.
COPY pyproject.toml ./
COPY src ./src
COPY arm ./arm
RUN pip install --no-deps .

# Bake the encoder the arm names, so a cold start needs no Hugging Face round-trip.
RUN python -c "import json; from sentence_transformers import SentenceTransformer; \
SentenceTransformer(json.load(open('/app/arm/meta.json'))['embedding_model'])"

EXPOSE 8000
CMD uvicorn router_service.api:app --host 0.0.0.0 --port ${PORT:-8000}
