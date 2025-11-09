import torch
from typing import Optional
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

_MODEL_NAME = "Salesforce/blip-image-captioning-base"
_device = "cuda" if torch.cuda.is_available() else "cpu"
_processor = None
_model = None

def _lazy_load():
    global _processor, _model
    if _processor is None or _model is None:
        _processor = BlipProcessor.from_pretrained(_MODEL_NAME)
        _model = BlipForConditionalGeneration.from_pretrained(_MODEL_NAME).to(_device)
        _model.eval()

@torch.inference_mode()
def generate_caption(pil: Image.Image, prompt: Optional[str] = None, max_new_tokens: int = 25) -> str:
    _lazy_load()
    inputs = _processor(images=pil, text=prompt or "", return_tensors="pt").to(_device)
    out = _model.generate(**inputs, max_new_tokens=max_new_tokens)
    cap = _processor.decode(out[0], skip_special_tokens=True)
    return cap.replace("aerial", "satellite")
