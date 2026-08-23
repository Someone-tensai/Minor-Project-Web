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

import history_store
from gradcam_utils import GradCam, get_target_layer, make_overlay
from model_config import discover_models, parse_model_spec
from model_loader import load_model

app = FastAPI(title="Grad-CAM Model Explorer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

history_store.init_db()

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
    return {
        "models": [
            {"name": name, "spec": cfg["spec"]} for name, cfg in registry.items()
        ]
    }


@app.post("/predict")
async def predict(
    model_name: str = Form(...),
    file: UploadFile = File(...),
    save_to_history: bool = Form(True),
):
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
    probabilities = {
        (class_names[i] if i < len(class_names) else f"class_{i}"): float(p)
        for i, p in enumerate(probs)
    }

    # Build the overlay from the resized (224x224) RGB image so it lines
    # up pixel-for-pixel with the cam map.
    display_img = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE)))
    overlay = make_overlay(display_img, cam_map)

    overlay_png = _encode_png_bytes(overlay)
    original_png = _encode_png_bytes(display_img)

    history_id = None
    if save_to_history:
        record = history_store.save_run(
            model_name=model_name,
            predicted_class=pred_label,
            predicted_index=int(pred_idx),
            probabilities=probabilities,
            original_png=original_png,
            overlay_png=overlay_png,
        )
        history_id = record["id"]

    return {
        "predicted_class": pred_label,
        "predicted_index": int(pred_idx),
        "probabilities": probabilities,
        "original_image": base64.b64encode(original_png).decode("utf-8"),
        "gradcam_overlay": base64.b64encode(overlay_png).decode("utf-8"),
        "history_id": history_id,
    }


@app.get("/history")
def get_history():
    runs = history_store.list_runs()
    return {
        "runs": [
            {
                "id": r["id"],
                "created_at": r["created_at"],
                "model_name": r["model_name"],
                "spec": parse_model_spec(r["model_name"]),
                "predicted_class": r["predicted_class"],
                "original_url": f"/history-images/{r['original_image_path']}",
                "overlay_url": f"/history-images/{r['overlay_image_path']}",
            }
            for r in runs
        ]
    }


@app.get("/history/{run_id}")
def get_history_item(run_id: str):
    record = history_store.get_run(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="No history entry with that id")
    return {
        "id": record["id"],
        "created_at": record["created_at"],
        "model_name": record["model_name"],
        "spec": parse_model_spec(record["model_name"]),
        "predicted_class": record["predicted_class"],
        "predicted_index": record["predicted_index"],
        "probabilities": record["probabilities"],
        "original_url": f"/history-images/{record['original_image_path']}",
        "overlay_url": f"/history-images/{record['overlay_image_path']}",
    }


@app.delete("/history/{run_id}")
def remove_history_item(run_id: str):
    deleted = history_store.delete_run(run_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="No history entry with that id")
    return {"deleted": run_id}


def _encode_png_bytes(rgb_array: np.ndarray) -> bytes:
    bgr = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    success, buf = cv2.imencode(".png", bgr)
    if not success:
        raise RuntimeError("Failed to encode image")
    return buf.tobytes()


# Serve saved history images and the frontend. Order matters: StaticFiles("/")
# is a catch-all mount, so more specific routes/mounts must come first.
app.mount(
    "/history-images",
    StaticFiles(directory=str(history_store.IMAGES_DIR)),
    name="history_images",
)
app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")


if __name__ == "__main__":
   import uvicorn
   uvicorn.run(app, host="0.0.0.0", port=8000)