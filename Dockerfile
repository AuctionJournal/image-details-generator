# syntax=docker/dockerfile:1

FROM python:3.11-slim

# ---- system deps ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libjpeg62-turbo libpng16-16 ca-certificates tesseract-ocr \
    build-essential cmake git && \
    rm -rf /var/lib/apt/lists/*

# ---- python deps ----
RUN python -m pip install --upgrade pip && \
    pip install \
      fastapi uvicorn[standard] \
      torch==2.4.1 torchvision==0.19.1 \
      transformers==4.44.2 tokenizers==0.19.1 \
      pillow requests pytesseract \
      llama-cpp-python==0.2.90 \
      huggingface_hub==0.24.6

# ---- model caches on image ----
ENV TRANSFORMERS_CACHE=/hf-cache \
    HF_HOME=/hf-cache
RUN mkdir -p /hf-cache /models /weights

# 1) Download BLIP-large (public) into /models
RUN python - <<'PY'\n\
from transformers import BlipProcessor, BlipForConditionalGeneration\n\
BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-large', cache_dir='/models')\n\
BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-large', cache_dir='/models')\n\
print('BLIP-large cached')\n\
PY

# 2) Download a public GGUF (Phi-3.5-mini-instruct Q4_K_M) into /weights
# If that repo ever requires a token, you can build with:  --build-arg HF_TOKEN=xxxx
ARG HF_TOKEN=""
ENV HUGGINGFACE_HUB_TOKEN=$HF_TOKEN
RUN python - <<'PY'\n\
from huggingface_hub import hf_hub_download\n\
p = hf_hub_download(repo_id='Qwen/Phi-3.5-mini-instruct-GGUF', filename='Phi-3.5-mini-instruct-Q4_K_M.gguf', local_dir='/weights', local_dir_use_symlinks=False)\n\
print('GGUF ->', p)\n\
PY

# ---- app ----
WORKDIR /app
COPY app.py /app/app.py

# env wiring
ENV BLIP_DIR=/models/Salesforce__blip-image-captioning-large \
    LLM_PATH=/weights/Phi-3.5-mini-instruct-Q4_K_M.gguf \
    MAX_ANALYZE_IMAGES=4

EXPOSE 8000
# gunicorn for multi-worker; WEB_CONCURRENCY env can tune
ENV WEB_CONCURRENCY=2
CMD ["bash", "-lc", "gunicorn -k uvicorn.workers.UvicornWorker -w ${WEB_CONCURRENCY:-2} -b 0.0.0.0:8000 app:app"]
