# src/geo_labeler_ext.py
from __future__ import annotations
import io, os, json, math, re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from PIL import Image, ExifTags
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform as rio_transform

import torch
from transformers import CLIPProcessor, CLIPModel

# ------------------------------ Configs ------------------------------

@dataclass
class LabelerConfig:
    classes: List[str]
    context: str = "This is a satellite image."
    clip_model_name: str = "openai/clip-vit-large-patch14"
    llm_threshold: float = 0.60  # 60%
    use_fast_preproc: bool = True
    # For captions (optional, if you plug a vision caption model)
    caption_prompt: str = "Describe this satellite image briefly and concretely."

# -------------------------- Utility functions ------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / (e.sum() + 1e-12)

def _normalize_im(arr: np.ndarray) -> np.ndarray:
    # clip & normalize per-channel to 0..255 uint8
    arr = np.nan_to_num(arr)
    out = []
    for c in range(arr.shape[0]):
        band = arr[c]
        lo, hi = np.percentile(band, [2, 98])
        if hi <= lo: hi = lo + 1
        band = np.clip((band - lo) / (hi - lo), 0, 1)
        out.append(band)
    out = (np.stack(out, 0) * 255).astype(np.uint8)
    return out

def _false_color_rgb(multiband: np.ndarray, rgb_bands: Tuple[int,int,int]) -> np.ndarray:
    # multiband: (C,H,W) -> select C indices (0-based)
    r, g, b = rgb_bands
    sel = multiband[[r, g, b]]
    return _normalize_im(sel)

def _pil_from_chw_uint8(arr_chw: np.ndarray) -> Image.Image:
    # arr: (C,H,W) uint8
    return Image.fromarray(np.transpose(arr_chw, (1,2,0)))

def _read_any_image_as_rgb(path: str) -> Tuple[Image.Image, Dict[str,Any]]:
    """
    Reads RGB or multispectral. If GeoTIFF with >3 bands, create a false-color composite (NIR,R,G).
    Returns PIL.Image (RGB) + metadata dict for geo-context extraction.
    """
    meta: Dict[str,Any] = {"source_path": path, "bands": None, "geoinfo": {}}
    ext = os.path.splitext(path)[1].lower()
    if ext in [".tif", ".tiff"]:
        with rasterio.open(path) as ds:
            meta["bands"] = ds.count
            meta["crs"] = str(ds.crs) if ds.crs else None
            meta["transform"] = tuple(ds.transform) if ds.transform else None
            meta["width"], meta["height"] = ds.width, ds.height
            arr = ds.read(out_dtype="float32")  # (C,H,W)
            # If multispectral (>=4), try NIR,R,G := bands (4,3,2) in many products.
            if arr.shape[0] >= 4:
                # 0-based indices: NIR=3, R=2, G=1  (common for Sentinel/Landsat stacks)
                rgb = _false_color_rgb(arr, (3,2,1))
            elif arr.shape[0] == 3:
                rgb = _normalize_im(arr)
            else:
                # single-band: replicate
                rgb = np.repeat(_normalize_im(arr)[0:1], 3, axis=0)
            pil = _pil_from_chw_uint8(rgb)
            return pil, meta
    else:
        # Regular RGB formats via PIL
        im = Image.open(path).convert("RGB")
        # EXIF for GPS
        exif = {}
        try:
            raw = im._getexif() or {}
            for k,v in raw.items():
                name = ExifTags.TAGS.get(k, k)
                exif[name] = v
        except Exception:
            pass
        meta["exif"] = exif
        meta["bands"] = 3
        return im, meta

def _extract_locality_from_meta(meta: Dict[str,Any]) -> Optional[str]:
    """
    Try GeoTIFF (rasterio) first, then EXIF GPS (PIL). No external reverse geocoding (works offline).
    Returns string like 'Lat, Lon (EPSG:4326)' or None.
    """
    # GeoTIFF with CRS + transform
    crs = meta.get("crs")
    transform = meta.get("transform")
    width, height = meta.get("width"), meta.get("height")
    if crs and transform and width and height:
        # Approximate center pixel -> geographic coords
        from affine import Affine
        A = Affine(*transform)
        cx, cy = width/2, height/2
        x, y = A * (cx, cy)
        try:
            src_crs = crs
            if "EPSG" not in src_crs:
                # best effort; rasterio handles strings like 'EPSG:4326'
                pass
            (lon, lat) = rio_transform(src_crs, "EPSG:4326", [x], [y])
            return f"{lat[0]:.5f}, {lon[0]:.5f} (EPSG:4326)"
        except Exception:
            return f"{y:.1f}, {x:.1f} ({crs})"

    # EXIF GPS
    exif = meta.get("exif") or {}
    gps = exif.get("GPSInfo")
    if isinstance(gps, dict):
        def _to_deg(v):
            num = lambda t: float(t[0]) / float(t[1])
            d = num(v[0]); m = num(v[1]); s = num(v[2])
            return d + m/60 + s/3600
        try:
            lat = _to_deg(gps[2]); lon = _to_deg(gps[4])
            if gps.get(1) == 'S': lat = -lat
            if gps.get(3) == 'W': lon = -lon
            return f"{lat:.5f}, {lon:.5f} (EXIF)"
        except Exception:
            pass

    # Fallback to filename hint
    name = os.path.basename(meta.get("source_path",""))
    m = re.search(r'(-?\d+\.\d+)[,_ ]+(-?\d+\.\d+)', name)
    if m:
        return f"{m.group(1)}, {m.group(2)} (filename)"
    return None

# ---------------------- Core labeler with CLIP -----------------------

class GeoVisionLabeler:
    """
    Drop-in classifier with:
    1) Confidence-aware multi-stage: LLM -> (if conf<0.6 or invalid) CLIP fallback.
    2) Multispectral hook: converts NIR/R/G to false-color RGB for vLLMs/CLIP.
    3) Auto geo-context retrieval from GeoTIFF/EXIF.
    4) Weak label JSON output ready for CSV export.
    5) Optional captions: use your vision caption model upstream and pass in; or plug later.
    """
    def __init__(self, cfg: LabelerConfig):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip = CLIPModel.from_pretrained(cfg.clip_model_name).to(self.device)
        self.clip_proc = CLIPProcessor.from_pretrained(cfg.clip_model_name)

        # Pre-encode class prompts for faster CLIP scoring
        self.class_prompts = [f"a satellite image of {c}" for c in cfg.classes]
        with torch.no_grad():
            inputs = self.clip_proc(text=self.class_prompts, return_tensors="pt", padding=True).to(self.device)
            self.class_text_emb = self.clip.get_text_features(**inputs)  # (K, D)
            self.class_text_emb = torch.nn.functional.normalize(self.class_text_emb, p=2, dim=-1)

    # --------- CLIP scoring used both for primary CLIP and LLM confidence -----

    def _clip_scores(self, pil_img: Image.Image) -> Tuple[np.ndarray, int, float]:
        self.clip.eval()
        with torch.no_grad():
            inputs = self.clip_proc(images=pil_img, return_tensors="pt").to(self.device)
            img_emb = self.clip.get_image_features(**inputs)  # (1,D)
            img_emb = torch.nn.functional.normalize(img_emb, p=2, dim=-1)  # (1,D)
            logits = (img_emb @ self.class_text_emb.T).squeeze(0)  # (K,)
            scores = logits.detach().cpu().float().numpy()
            probs = _softmax(scores)
            idx = int(np.argmax(probs))
            conf = float(probs[idx])
            return probs, idx, conf

    def _estimate_llm_confidence(self, pil_img: Image.Image, llm_label: str) -> float:
        """
        Confidence proxy: Use CLIP to re-score the image against class prompts and
        take the probability of the LLM-chosen class. Works offline & model-agnostic.
        """
        llm_label = llm_label.strip()
        if llm_label not in self.cfg.classes:
            return 0.0
        probs, idx, _ = self._clip_scores(pil_img)
        cls_idx = self.cfg.classes.index(llm_label)
        return float(probs[cls_idx])

    # ------------------------ Public API --------------------------------------

    def classify_image(
        self,
        image_path: str,
        llm_pred_label: Optional[str] = None,    # If you run an LLM mapper upstream, pass its label here
        description: Optional[str] = None        # If you run a vision captioner upstream, pass caption here
    ) -> Dict[str, Any]:
        """
        Returns weak-label JSON:
        {
          "filename": ...,
          "predicted_label": ...,
          "confidence": 0.0..1.0,
          "description": "...",
          "locality": "lat, lon (source)",
          "mode": "llm" or "clip_fallback",
        }
        """
        pil, meta = _read_any_image_as_rgb(image_path)
        locality = _extract_locality_from_meta(meta)

        # If we have an LLM prediction, check confidence + validity; else go straight to CLIP.
        if llm_pred_label:
            conf_est = self._estimate_llm_confidence(pil, llm_pred_label)
            if (llm_pred_label not in self.cfg.classes) or (conf_est < self.cfg.llm_threshold):
                # Fallback to CLIP
                probs, idx, conf = self._clip_scores(pil)
                label = self.cfg.classes[idx]
                mode = "clip_fallback"
            else:
                label = llm_pred_label
                conf = conf_est
                mode = "llm"
        else:
            # No LLM given -> use CLIP directly
            probs, idx, conf = self._clip_scores(pil)
            label = self.cfg.classes[idx]
            mode = "clip_only"

        result = {
            "filename": os.path.basename(image_path),
            "predicted_label": label,
            "confidence": round(float(conf), 4),
            "description": description or "",  # you can fill via your caption model upstream
            "locality": locality or "",
            "mode": mode,
            "classes": self.cfg.classes,  # handy for auditing
        }
        return result

# ------------------ Notes on multispectral/vLLM support -----------------------

"""
Multi-Spectral/Non-RGB Support (what changes):
- Most vision-LLMs (and CLIP) expect 3-channel RGB. For 4+ bands (e.g., NIR), we create
  a false-color RGB composite (NIR,R,G) inside _read_any_image_as_rgb(). This lets you
  reuse the same vision model while still leveraging NIR signal (common remote-sensing practice).
- Preprocessing already normalizes per-band and converts to uint8 RGB.

If you control a custom vLLM that supports arbitrary channels:
- Replace _read_any_image_as_rgb() to return a multi-channel tensor and adapt your vLLM
  preprocessor to accept C>3.
- Optionally compute indices (NDVI = (NIR - R)/(NIR + R)) and concatenate as an extra channel,
  or summarize as a numeric feature you append to the text context (e.g., context+=f" NDVI={ndvi:.2f}").

Limitations (acknowledged):
- Off-the-shelf vLLMs/CLIP are trained on RGB; multi-spectral input must be projected to 3 channels
  or you must use a specialized model.
"""
