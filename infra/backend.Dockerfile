FROM python:3.11-slim

# Install system dependencies (FFmpeg for audio processing)
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch CPU-only first (large layer, cached separately)
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install Python dependencies
COPY server/requirements.txt /app/server/requirements.txt
RUN pip install --no-cache-dir -r server/requirements.txt

# Install additional ML dependencies not in requirements.txt
RUN pip install --no-cache-dir librosa matplotlib numpy

# Copy application code
COPY config.py /app/config.py
COPY feature_extractor.py /app/feature_extractor.py
COPY model.py /app/model.py
COPY server/ /app/server/

# Copy and set up startup script (strip Windows CRLF line endings)
COPY infra/start_backend.sh /app/start_backend.sh
RUN sed -i 's/\r$//' /app/start_backend.sh && chmod +x /app/start_backend.sh

# Create models directory
RUN mkdir -p /app/models

ENV VOCALUITY_BASE_DIR=/app
ENV PORT=8000

EXPOSE 8000

CMD ["/app/start_backend.sh"]
