"""
Models are auto-discovered from MODELS_DIR using a key=value filename scheme:

  arch-{resnet18|resnet50|vgg16}
  _train-{federated|centralized}
  _freeze-{layer4in|layer4out|frozen|fullunfreeze}      (optional)
  _dist-{iid|noniid}                                     (federated only)
  _strat-{fedavg|fedprox}                                (federated only)
  _clients-{int}                                         (federated only)
  _rounds-{int}
  _lr-{default|lrr|<number>}                             (optional)
  _mu-{number}                                           (fedprox only)
  _aug-1                                                 (optional, omitted if not augmented)
  _date-{YYYY-MM-DD}
  _tag-{freeform}                                        (optional, always last)

e.g. arch-resnet18_train-federated_freeze-layer4in_dist-noniid_strat-fedprox_clients-5_rounds-20_lr-default_mu-0.01_date-2026-08-23.pth

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
STRATEGY_LABELS = {"fedavg": "FedAvg", "fedprox": "FedProx"}
TRAIN_LABELS = {"federated": "Federated", "centralized": "Centralized"}
FREEZE_LABELS = {
    "layer4in": "Layer4 (in)",
    "layer4out": "Layer4 (out)",
    "frozen": "Frozen",
    "fullunfreeze": "Full unfreeze",
}
DIST_LABELS = {"iid": "IID", "noniid": "Non-IID"}
LR_LABELS = {"default": "Default", "lrr": "LRR"}


def parse_kv_spec(stem: str):
    """Parse the new arch-...-key-value... naming scheme. Returns None if
    the filename doesn't start with 'arch-' (i.e. it's an old-scheme file)."""
    if not stem.startswith("arch-"):
        return None

    fields = {}
    for chunk in stem.split("_"):
        if "-" not in chunk:
            continue
        key, val = chunk.split("-", 1)
        fields[key] = val

    arch = fields.get("arch")
    train = fields.get("train")
    freeze = fields.get("freeze")
    dist = fields.get("dist")
    strat = fields.get("strat")
    clients = fields.get("clients")
    rounds = fields.get("rounds")
    lr = fields.get("lr")
    mu = fields.get("mu")
    aug = fields.get("aug")
    date = fields.get("date")
    tag = fields.get("tag")

    return {
        "training_type": train,
        "arch": ARCH_LABELS.get(arch, arch),
        "layer": FREEZE_LABELS.get(freeze, freeze) if freeze else None,
        "distribution": DIST_LABELS.get(dist, dist) if dist else None,
        "strategy": STRATEGY_LABELS.get(strat, strat) if strat else None,
        "clients": int(clients) if clients else None,
        "rounds": int(rounds) if rounds else None,
        "lr": LR_LABELS.get(lr, lr) if lr else None,
        "mu": float(mu) if mu else None,
        "augmented": aug == "1",
        "date": date,
        "tag": tag,
        # raw (unlabeled) values -- used by the frontend cascade filter,
        # since it needs to match dropdown values, not display labels
        "_raw": {
            "arch": arch, "train": train, "freeze": freeze, "dist": dist,
            "strat": strat, "clients": clients, "rounds": rounds,
            "lr": lr, "mu": mu, "aug": aug, "date": date, "tag": tag,
        },
    }


def parse_legacy_spec(stem: str) -> dict:
    """Old dash-based parser, kept only so history entries recorded under
    old filenames (before the rename) still display correctly."""
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

    mu_match = re.search(r"(?<![\d.])(0\.\d+)(?![\d.])", stem)
    mu = float(mu_match.group(1)) if (mu_match and strategy == "fedprox") else None

    augmented = "augmented" in lower_tokens

    date_match = re.search(r"(\d{4}-\d{2}-\d{2})", stem)
    date = date_match.group(1) if date_match else None

    return {
        "training_type": "federated" if strategy else None,
        "arch": ARCH_LABELS.get(arch, arch),
        "layer": layer,
        "distribution": distribution,
        "strategy": STRATEGY_LABELS.get(strategy, strategy) if strategy else None,
        "clients": clients,
        "rounds": rounds,
        "lr": None,
        "mu": mu,
        "augmented": augmented,
        "date": date,
        "tag": None,
        "_raw": None,
    }


def parse_model_spec(stem: str) -> dict:
    """Try the new key=value scheme first; fall back to the legacy parser
    for old filenames still referenced by existing history entries."""
    parsed = parse_kv_spec(stem)
    return parsed if parsed is not None else parse_legacy_spec(stem)


def discover_models() -> dict:
    """Scan MODELS_DIR for .pth files and build the config dict on the fly."""
    registry = {}
    if not MODELS_DIR.exists():
        return registry

    for pth_file in sorted(MODELS_DIR.glob("*.pth")):
        stem = pth_file.stem
        spec = parse_kv_spec(stem)
        if spec is None:
            print(f"[warn] skipping {pth_file.name}: doesn't match the "
                  f"arch-..._train-..._..._date-... naming scheme")
            continue

        raw_arch = spec["_raw"]["arch"]
        if raw_arch not in KNOWN_ARCHS:
            print(f"[warn] skipping {pth_file.name}: unrecognized arch {raw_arch!r}")
            continue

        registry[stem] = {
            "file": str(pth_file),
            "arch": raw_arch,
            "num_classes": NUM_CLASSES,
            "class_names": CLASS_NAMES,
            "spec": spec,
        }

    return registry
