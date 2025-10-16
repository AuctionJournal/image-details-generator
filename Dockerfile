FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libjpeg62-turbo libpng16-16 ca-certificates tesseract-ocr \
    build-essential cmake curl git \
 && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip \
 && pip install \
      fastapi uvicorn[standard] \
      torch==2.4.1 torchvision==0.19.1 \
      transformers==4.44.2 tokenizers==0.19.1 \
      pillow requests pytesseract \
      llama-cpp-python==0.2.90

WORKDIR /app
COPY app.py /app/app.py

RUN python - <<'PY'\n\
from huggingface_hub import snapshot_download\n\
snapshot_download(repo_id="Salesforce/blip-image-captioning-large",\n\
                 local_dir="/app/models/Salesforce__blip-image-captioning-large",\n\
                 allow_patterns=["*.json","*.bin","*.txt","*.model","*.py","*.safetensors"],\n\
                 local_dir_use_symlinks=False)\n\
PY

RUN mkdir -p /llm && \
    curl -L -o /llm/Phi-3.5-mini-instruct-Q4_K_M.gguf \
    https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf

ENV BLIP_DIR=/app/models/Salesforce__blip-image-captioning-large
ENV LLM_PATH=/llm/Phi-3.5-mini-instruct-Q4_K_M.gguf
ENV LLM_N_GPU_LAYERS=0
ENV HOST=0.0.0.0 PORT=8000

EXPOSE 8000
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8000","--workers","1"]
