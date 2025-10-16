import os, io, re, json, time
from typing import List, Dict, Tuple
from urllib.parse import urlparse

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from PIL import Image, ImageOps, ImageFile
import pytesseract
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from llama_cpp import Llama

# ---------- env ----------
CATEGORIES_URL = os.environ.get(
    "CATEGORIES_URL",
    "https://aj-backend-latest.azurewebsites.net/api/lotCategory/fetchAllLotCategories",
)
BLIP_DIR = os.environ.get("BLIP_DIR", "/models/Salesforce__blip-image-captioning-large")
LLM_PATH = os.environ.get("LLM_PATH", "/weights/phi-3.5-mini-instruct-Q4_K_M.gguf")
MAX_IMAGES = int(os.environ.get("MAX_ANALYZE_IMAGES", "4"))
LLM_N_GPU_LAYERS = int(os.environ.get("LLM_N_GPU_LAYERS", "0"))

# ---------- torch/PIL setup ----------
ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.set_grad_enabled(False)
torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# ---------- models (lazy, module singletons) ----------
_blip_proc = _blip_model = _llm = None
_id_to_name: Dict[int, str] = {}
_other_id: int = 0
_last_cat_fetch = 0.0

def ensure_models_loaded():
    global _blip_proc, _blip_model, _llm
    if _blip_proc is None or _blip_model is None:
        _blip_proc = BlipProcessor.from_pretrained(BLIP_DIR, local_files_only=True)
        _blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_DIR, local_files_only=True).to(DEVICE).eval()
    if _llm is None:
        _llm = Llama(model_path=LLM_PATH, n_ctx=4096, n_threads=os.cpu_count() or 4, n_gpu_layers=LLM_N_GPU_LAYERS)

# ---------- categories ----------
# --- make category fetch fail-soft instead of crashing the app ---
def fetch_categories(force=False) -> Tuple[Dict[int, str], int]:
    global _id_to_name, _other_id, _last_cat_fetch
    if not force and time.time() - _last_cat_fetch < 300 and _id_to_name:
        return _id_to_name, _other_id
    try:
        r = requests.post(CATEGORIES_URL, json={"isFindLotCount": True}, timeout=10)
        r.raise_for_status()
        data = (r.json() or {}).get("data") or []
    except Exception:
        # fallback: keep previous cache or minimal OTHER set
        if _id_to_name:
            return _id_to_name, _other_id
        _id_to_name, _other_id = {0: "OTHER"}, 0
        _last_cat_fetch = time.time()
        return _id_to_name, _other_id

    id_to_name = {}
    other_id = None
    for it in data:
        try:
            cid = int(it.get("categoryID"))
            name = str(it.get("categoryName") or "").strip()
        except Exception:
            continue
        if name:
            id_to_name[cid] = name
            if other_id is None and name.upper() == "OTHER":
                other_id = cid
    if other_id is None:
        other_id = max(id_to_name.keys()) if id_to_name else 0
    _id_to_name, _other_id, _last_cat_fetch = id_to_name, other_id, time.time()
    return _id_to_name, _other_id

# ---------- image utils ----------
def _load_image(src: str) -> Image.Image:
    if urlparse(src).scheme in ("http", "https"):
        resp = requests.get(src, timeout=30)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    return Image.open(src).convert("RGB")

def _downscale_for_caption(img: Image.Image, max_side=1024, q=85) -> bytes:
    w, h = img.size
    if max(w, h) > max_side:
        s = max_side / float(max(w, h))
        img = img.resize((int(w * s), int(h * s)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=q, optimize=True)
    return buf.getvalue()

def _ocr_text(img: Image.Image) -> str:
    g = ImageOps.autocontrast(ImageOps.grayscale(img))
    return pytesseract.image_to_string(g, lang="eng").strip()

def _ocr_has_conf(t: str) -> bool:
    toks = [w for w in t.split() if any(c.isalnum() for c in w)]
    if not toks:
        return False
    avg = sum(len(w) for w in toks) / len(toks)
    return len(toks) >= 5 and avg >= 3

def _caption_from_bytes(jpeg: bytes) -> str:
    pil = Image.open(io.BytesIO(jpeg)).convert("RGB")
    inp = _blip_proc(pil, return_tensors="pt").to(DEVICE)
    out = _blip_model.generate(**inp, max_new_tokens=64)
    return _blip_proc.decode(out[0], skip_special_tokens=True).strip()

# ---------- LLM composer ----------
def compose_with_llm(caps: List[str], ocrs: List[str], categories_text: str):
    sys = (
        "You write concise ecommerce copy from multiple product photos.\n"
        "Use evidence from ALL images and any OCR text provided.\n"
        "Choose a categoryID ONLY from the provided list.\n"
        "Return EXACTLY three lines:\n"
        "Title: <<= 8 words>\n"
        "Description: <1–2 factual sentences>\n"
        "CategoryID: <number from the list>\n"
        "No markdown or extra text."
    )
    cap_block = "\n".join(f"{i+1}. {t}" for i, t in enumerate(caps))
    ocr_clean = [t.strip() for t in ocrs if t and t.strip()]
    ocr_block = "\n".join(f"{i+1}. {t}" for i, t in enumerate(ocr_clean)) or "(none)"
    usr = (
        "CATEGORIES (ID - Name):\n"
        f"{categories_text}\n\n"
        "CAPTIONS:\n"
        f"{cap_block}\n\n"
        "OCR TEXT:\n"
        f"{ocr_block}\n\n"
        "Respond with exactly three lines:\nTitle:\nDescription:\nCategoryID:"
    )
    prompt = f"<s>[INST] <<SYS>>\n{sys}\n<</SYS>>\n{usr}\n[/INST]"
    out = _llm(prompt, max_tokens=180, temperature=0.2, top_p=0.9, repeat_penalty=1.05)
    txt = out["choices"][0]["text"].strip()

    title = desc = ""
    cat = None
    for line in txt.splitlines():
        l = line.strip()
        if not title and l.lower().startswith("title"):
            title = re.sub(r"^title\s*:\s*", "", l, flags=re.I).strip()
        elif not desc and l.lower().startswith("description"):
            desc = re.sub(r"^description\s*:\s*", "", l, flags=re.I).strip()
        elif cat is None and l.lower().startswith("categoryid"):
            num = re.sub(r"^categoryid\s*:\s*", "", l, flags=re.I).strip()
            try:
                cat = int(re.findall(r"-?\d+", num)[0])
            except Exception:
                pass

    if not title or not desc:
        lines = [l for l in txt.splitlines() if l.strip()]
        if lines:
            title = title or lines[0][:80]
            desc = desc or " ".join(lines[1:]).strip()
    if len(title.split()) > 8:
        title = " ".join(title.split()[:8])
    if desc and not desc.endswith("."):
        desc += "."

    return title, desc, cat

# ---------- API ----------
class AnalyzeIn(BaseModel):
    images: List[str] = Field(..., min_items=1, max_items=8)
    max_images: int = Field(default=4, ge=1, le=8)

class AnalyzeOut(BaseModel):
    title: str
    description: str
    categoryID: int
    categoryName: str

app = FastAPI(title="image-details-generator")

@app.on_event("startup")
def _startup():
    pass

@app.post("/analyze", response_model=AnalyzeOut)
def analyze(body: AnalyzeIn):
    ensure_models_loaded()
    id2name, other_id = fetch_categories()
    allowed = set(id2name.keys())
    cats_text = "\n".join(f"{k} - {v}" for k, v in sorted(id2name.items()))

    caps, ocrs = [], []
    for src in body.images[: min(body.max_images, MAX_IMAGES)]:
        try:
            img = _load_image(src)
            jpeg = _downscale_for_caption(img, max_side=1024, q=85)
            caps.append(_caption_from_bytes(jpeg))
            quick = _ocr_text(Image.open(io.BytesIO(jpeg)))
            if _ocr_has_conf(quick):
                ocrs.append(quick)
            else:
                # try higher-res
                w, h = img.size
                if max(w, h) < 1800:
                    s = 1800 / float(max(w, h))
                    img = img.resize((int(w * s), int(h * s)), Image.Resampling.LANCZOS)
                ocrs.append(_ocr_text(img) or quick)
        except Exception as e:
            caps.append(f"(failed to read {src}: {e})")
            ocrs.append("")

    title, desc, cat = compose_with_llm(caps, ocrs, cats_text)
    if cat not in allowed:
        cat = other_id
    return AnalyzeOut(
        title=title, description=desc, categoryID=cat, categoryName=id2name.get(cat, "OTHER")
    )

# --- make /healthz indicate readiness but never block on models or network ---
@app.get("/healthz")
def healthz():
    ready = _blip_model is not None and _llm is not None and bool(_id_to_name)
    return {"ok": True, "ready": ready}
