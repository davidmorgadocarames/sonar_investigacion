FROM python:3.11-slim

# CPU by default; compose.gpu.yaml overrides it with a CUDA index.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN useradd --create-home --uid 1000 app

WORKDIR /app

COPY requirements.txt .
RUN pip install --extra-index-url "${TORCH_INDEX_URL}" -r requirements.txt

COPY src/ src/
COPY tests/ tests/
RUN mkdir -p datasets models outputs && chown -R app:app /app

USER app

ENTRYPOINT ["python", "-m", "sonar_pipeline"]
CMD ["--help"]
