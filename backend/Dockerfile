FROM python:3.10-slim

WORKDIR /app

# Runtime libs commonly required by OpenCV video decoding.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libglib2.0-0 libgl1 libgles2 libegl1 \
    && rm -rf /var/lib/apt/lists/*

# Cloud default: force CPU unless explicitly enabled and runtime-ready.
ENV MEDIAPIPE_USE_GPU=0

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --prefer-binary -r /app/requirements.txt

COPY . /app

# WeChat Cloud Hosting injects PORT. Fallback to 8000 for local runs.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
