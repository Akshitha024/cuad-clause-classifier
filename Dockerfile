# CPU-only runtime (no CUDA layers). Build pulls ~2GB of torch wheels.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/work/.hf-cache

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install .

ENTRYPOINT ["clause-x"]
CMD ["--help"]
