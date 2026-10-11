FROM python:3.11.17-slim

# CPU by default; compose.gpu.yaml overrides it with the CUDA 13.0 index (D-019). Only
# requirements-torch.txt is installed from this index; requirements.txt comes from PyPI alone.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN useradd --create-home --uid 1000 app

WORKDIR /app

COPY requirements-torch.txt .
RUN pip install --index-url "${TORCH_INDEX_URL}" -r requirements-torch.txt

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ src/
COPY tests/ tests/
COPY tools/ tools/
RUN mkdir -p datasets models outputs && chown -R app:app /app

USER app

ENTRYPOINT ["python", "-m", "sonar_pipeline"]
CMD ["--help"]
