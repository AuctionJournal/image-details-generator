# CPU build for Azure App Service (Premium V2/V3)
FROM python:3.11-slim

# ---- system libs: OCR + build tools for llama-cpp-python ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libjpeg62-turbo libpng16-16 ca-certificates tesseract-ocr \
    build-essential cmake git curl \
 && rm -rf /var/lib/apt/lists/*

# ---- python deps ----
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir \
    fastapi uvicorn[standard] \
    torch==2.4.1 torchvision==0.19.1 \
    transformers==4.44.2 tokenizers==0.19.1 \
    pillow requests pytesseract huggingface_hub==0.23.0 \
    llama-cpp-python==0.2.90

# ---- app files ----
WORKDIR /app
COPY app.py /app/app.py

# ---- download models during build ----
# BLIP (public)
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download( \
    repo_id='Salesforce/blip-image-captioning-large', \
    local_dir='/app/models/Salesforce__blip-image-captioning-large', \
    allow_patterns=['*.json','*.bin','*.txt','*.model','*.py','*.safetensors'], \
    local_dir_use_symlinks=False)"

# Phi-3.5-mini-instruct GGUF (public mirror)
RUN mkdir -p /llm && \
    curl -L -o /llm/Phi-3.5-mini-instruct-Q4_K_M.gguf \
    https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf

# ---- defaults (override in App Service Configuration if you want) ----
ENV BLIP_DIR=/app/models/Salesforce__blip-image-captioning-large
ENV LLM_PATH=/llm/Phi-3.5-mini-instruct-Q4_K_M.gguf
ENV LLM_N_GPU_LAYERS=0

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
