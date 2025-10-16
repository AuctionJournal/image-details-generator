FROM python:3.11-slim

# --- system deps: OCR + build chain + BLAS for llama-cpp ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libjpeg62-turbo libpng16-16 ca-certificates tesseract-ocr \
    build-essential cmake ninja-build curl git \
    libopenblas-dev \
 && rm -rf /var/lib/apt/lists/*

# --- python base ---
RUN python -m pip install --upgrade pip setuptools wheel

# --- app deps (CPU) ---
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir \
      fastapi uvicorn[standard] \
      pillow<11 requests pytesseract \
      "huggingface_hub>=0.23,<0.25" \
      "transformers==4.44.2" \
      "tokenizers==0.19.1" \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      "torch==2.4.1+cpu" "torchvision==0.19.1+cpu" && \
    pip install --no-cache-dir "llama-cpp-python==0.2.90"
# Force llama-cpp to use OpenBLAS on CPU; prefer wheels but can build if needed
ENV CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"
ENV FORCE_CMAKE=1
RUN pip install --no-cache-dir --prefer-binary llama-cpp-python==0.2.90

# --- app files ---
WORKDIR /app
COPY app.py /app/app.py

# --- download models at build time ---
# BLIP weights
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download( \
    repo_id='Salesforce/blip-image-captioning-large', \
    local_dir='/app/models/Salesforce__blip-image-captioning-large', \
    allow_patterns=['*.json','*.bin','*.txt','*.model','*.py','*.safetensors'], \
    local_dir_use_symlinks=False)"

# Phi GGUF
RUN mkdir -p /llm && \
    curl -L -o /llm/Phi-3.5-mini-instruct-Q4_K_M.gguf \
    https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf

# --- runtime env ---
ENV BLIP_DIR=/app/models/Salesforce__blip-image-captioning-large
ENV LLM_PATH=/llm/Phi-3.5-mini-instruct-Q4_K_M.gguf
ENV LLM_N_GPU_LAYERS=0

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
