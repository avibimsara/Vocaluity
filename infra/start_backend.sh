#!/bin/sh
set -e

# Download model files from GCS if MODEL_BUCKET is set (non-fatal — server works with random weights)
if [ -n "$MODEL_BUCKET" ]; then
  echo "[startup] Downloading models from gs://${MODEL_BUCKET}/models/ ..."
  python -c "
from google.cloud import storage
import os

bucket_name = os.environ['MODEL_BUCKET']
client = storage.Client()
bucket = client.bucket(bucket_name)
blobs = list(bucket.list_blobs(prefix='models/'))

count = 0
for blob in blobs:
    if blob.name.endswith('.pth'):
        dest = f'/app/{blob.name}'
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        blob.download_to_filename(dest)
        print(f'  Downloaded {blob.name} -> {dest}')
        count += 1

if count == 0:
    print('[startup] No .pth files found in bucket, will use random weights.')
else:
    print(f'[startup] Downloaded {count} model file(s).')
" || echo "[startup] WARNING: Model download failed, continuing with random weights."
else
  echo "[startup] MODEL_BUCKET not set, skipping model download."
fi

echo "[startup] Starting uvicorn..."
exec uvicorn server.main:app --host 0.0.0.0 --port "${PORT:-8000}"
