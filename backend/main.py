import base64
import io

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision import transforms

from gradcam_utils import GradCam, get_target_layer, make_overlay
from model_config import discover_models
from model_loader import load_model

app = FastAPI(title="Grad-CAM Model Explorer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Standard ImageNet preprocessing. our_model() initializes from
# torchvision's pretrained ImageNet weights before fine-tuning, so the
# backbone expects inputs normalized this way.
# ---------------------------------------------------------------------------
IMG_SIZE = 224
PREPROCESS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_MODEL_CACHE: dict = {}  # model_name -> (nn.Module, GradCam)


def get_cached(model_name: str):
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    registry = discover_models()
    if model_name not in registry:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_name}")

    cfg = registry[model_name]
    model = load_model(cfg["file"], cfg["arch"], cfg["num_classes"])
    target_layer = get_target_layer(model, cfg["arch"])
    cam = GradCam(model, target_layer)

    _MODEL_CACHE[model_name] = (model, cam, cfg)
    return _MODEL_CACHE[model_name]


@app.get("/models")
def list_models():
    registry = discover_models()
    return {"models": list(registry.keys())}


@app.post("/predict")
async def predict(model_name: str = Form(...), file: UploadFile = File(...)):
    model, cam, cfg = get_cached(model_name)

    raw_bytes = await file.read()
    pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")

    input_tensor = PREPROCESS(pil_img).unsqueeze(0)  # [1, 3, 224, 224]

    cam_map, pred_idx, logits = cam.generate(input_tensor)
    probs = F.softmax(logits, dim=1).squeeze().detach().cpu().numpy()

    class_names = cfg["class_names"]
    pred_label = (
        class_names[pred_idx] if pred_idx < len(class_names) else f"class_{pred_idx}"
    )

    # Build the overlay from the resized (224x224) RGB image so it lines
    # up pixel-for-pixel with the cam map.
    display_img = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE)))
    overlay = make_overlay(display_img, cam_map)

    overlay_b64 = _encode_png(overlay)
    original_b64 = _encode_png(display_img)

    return {
        "predicted_class": pred_label,
        "predicted_index": int(pred_idx),
        "probabilities": {
            (class_names[i] if i < len(class_names) else f"class_{i}"): float(p)
            for i, p in enumerate(probs)
        },
        "original_image": original_b64,
        "gradcam_overlay": overlay_b64,
    }


def _encode_png(rgb_array: np.ndarray) -> str:
    bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    success, buf = cv2.imencode(".png", bgr)
    if not success:
        raise RuntimeError("Failed to encode image")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


# Serve the frontend directly from the backend so this whole thing is one
# process: python main.py -> open http://localhost:8000
app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")
