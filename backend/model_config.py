"""
Auto-discovers models: drop any .pth file into backend/models/ and it'll
show up in the dropdown automatically, as long as its filename STARTS
with one of "resnet18", "resnet50", or "vgg16" (which yours already do,
e.g. resnet18-layer4_in-iid-fedavg-3clients-20--Default--2026-08-20.pth).

Edit CLASS_NAMES below to your real class labels, in the correct order.
All models were confirmed to be trained with num_classes=4.
"""

import re
from pathlib import Path

MODELS_DIR = Path(__file__).parent / "models"

NUM_CLASSES = 4
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

KNOWN_ARCHS = ["resnet18", "resnet50", "vgg16"]
KNOWN_STRATEGIES = ["fedavg", "fedprox"]

ARCH_LABELS = {"resnet18": "ResNet18", "resnet50": "ResNet50", "vgg16": "VGG16"}
STRATEGY_LABELS = {"fedavg": "FedAvg", "fedprox": "FedProx", "centralized": "Centralized"}


def parse_model_spec(stem: str) -> dict:
    """
    Model filenames double as experiment records, e.g.:
      resnet18-layer4_out-noniid-fedprox-3clients-20--Default-0.01-2026-08-20

    Rather than assume a fixed field order (the dashes aren't consistent --
    some fields are blank, e.g. "--Default--"), this pulls known tokens out
    by pattern match wherever they appear. Unrecognized tokens are ignored
    rather than breaking the parse.
    """
    tokens = [t for t in re.split(r"[-_]", stem) if t]
    lower_tokens = [t.lower() for t in tokens]
    joined = stem.lower()

    arch = next((a for a in KNOWN_ARCHS if lower_tokens and lower_tokens[0] == a), None)
    strategy = next((s for s in KNOWN_STRATEGIES if s in lower_tokens), None)

    layer_match = re.search(r"layer(\d+)_(in|out)", joined)
    layer = f"Layer{layer_match.group(1)} ({layer_match.group(2)})" if layer_match else None

    if "noniid" in lower_tokens:
        distribution = "Non-IID"
    elif "iid" in lower_tokens:
        distribution = "IID"
    else:
        distribution = None

    clients_match = re.search(r"(\d+)clients?", joined)
    clients = int(clients_match.group(1)) if clients_match else None

    # Rounds: a bare integer token that isn't the client count, appearing
    # after the strategy token.
    rounds = None
    if strategy:
        try:
            strat_pos = lower_tokens.index(strategy)
            for t in lower_tokens[strat_pos + 1:]:
                if t.isdigit():
                    rounds = int(t)
                    break
        except ValueError:
            pass

    # FedProx proximal term mu: a decimal-looking token, e.g. "0.01"
    mu_match = re.search(r"(?<![\d.])(0\.\d+)(?![\d.])", stem)
    mu = float(mu_match.group(1)) if (mu_match and strategy == "fedprox") else None

    augmented = "augmented" in lower_tokens

    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", stem)
    date = date_match.group(1) if date_match else None

    return {
        "arch": ARCH_LABELS.get(arch, arch),
        "layer": layer,
        "distribution": distribution,
        "strategy": STRATEGY_LABELS.get(strategy, strategy) if strategy else None,
        "clients": clients,
        "rounds": rounds,
        "mu": mu,
        "augmented": augmented,
        "date": date,
    }


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
            "spec": parse_model_spec(stem),
        }

    return registry
