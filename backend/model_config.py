"""
Auto-discovers models: drop any .pth file into backend/models/ and it'll
show up in the dropdown automatically, as long as its filename STARTS
with one of "resnet18", "resnet50", or "vgg16" (which yours already do,
e.g. resnet18-layer4_in-iid-fedavg-3clients-20--Default--2026-08-20.pth).

Edit CLASS_NAMES below to your real class labels, in the correct order.
All models were confirmed to be trained with num_classes=4.
"""

from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"

NUM_CLASSES = 4
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

KNOWN_ARCHS = ["resnet18", "resnet50", "vgg16"]


def discover_models() -> dict:
    """Scan MODELS_DIR for .pth files and build the config dict on the fly."""
    registry = {}
    if not MODELS_DIR.exists():
        return registry

    for pth_file in sorted(MODELS_DIR.glob("*.pth")):
        stem = pth_file.stem
        arch = next((a for a in KNOWN_ARCHS if stem.startswith(a)), None)
        if arch is None:
            print(f"[warn] skipping {pth_file.name}: filename doesn't start "
                  f"with a known arch ({KNOWN_ARCHS})")
            continue

        registry[stem] = {
            "file": str(pth_file),
            "arch": arch,
            "num_classes": NUM_CLASSES,
            "class_names": CLASS_NAMES,
        }

    return registry
