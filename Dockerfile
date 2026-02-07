FROM python:3.12-slim

# System deps for librosa / soundfile / audio decoding
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

RUN mkdir -p /tmp/chord-uploads

# Cache numba JIT compilation at build time (avoids 60s+ cold-start)
ENV NUMBA_CACHE_DIR=/app/.numba_cache
RUN mkdir -p /app/.numba_cache && \
    python -c "\
import numpy as np; \
import librosa; \
y = np.random.randn(22050).astype(np.float32); \
librosa.feature.chroma_cqt(y=y, sr=22050, hop_length=512); \
print('numba cache warmed')"

# Railway injects PORT env var, default 5000 for local
ENV PORT=5000

EXPOSE ${PORT}

# --preload: load app once before forking (shared memory)
# --workers 1: Railway free/hobby tier has limited RAM
# --timeout 300: long audio files need processing time
CMD gunicorn --bind 0.0.0.0:${PORT} --workers 1 --preload --timeout 300 --graceful-timeout 30 app:app
