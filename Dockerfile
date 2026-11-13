FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NUMBA_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MPLCONFIGDIR=/tmp/spatial-matplotlib \
    NUMBA_CACHE_DIR=/tmp/spatial-numba

WORKDIR /workspace
COPY requirements.txt ./
RUN python -m pip install --only-binary=:all: --index-url https://download.pytorch.org/whl/cpu "torch>=2.6,<3" \
    && python -m pip install --only-binary=:all: -r requirements.txt

COPY config ./config
COPY src ./src
COPY app ./app

# Bake deterministic offline demo snapshots once; both services read the same image.
RUN python -m src.data_loader --mode synthetic \
    && python -m src.preprocessing \
    && python -m src.models.spatial_gat \
    && useradd --create-home --uid 10001 spatial \
    && chown -R spatial:spatial /workspace /tmp/spatial-matplotlib /tmp/spatial-numba

USER spatial
EXPOSE 8000 8501
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
