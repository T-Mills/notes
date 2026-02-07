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

# Railway injects PORT env var, default 5000 for local
ENV PORT=5000

EXPOSE ${PORT}

# 300s timeout handles long audio files; workers=2 keeps RAM sane
CMD gunicorn --bind 0.0.0.0:${PORT} --workers 2 --timeout 300 --graceful-timeout 30 app:app
