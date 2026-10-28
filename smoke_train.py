import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

IN_NOTEBOOK = "ipykernel" in sys.modules
if not IN_NOTEBOOK:
    matplotlib.use("Agg")  # run as a script: save figures, never open windows

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from roboflow import Roboflow
from ultralytics import YOLO

from api_key import load_api_key
from costing import (alternatives, best_band, expected_cost, outcome_rates, per_thousand,
                     review_share, reviewers_needed, threshold_sweep)
from tiling import GRID, IMAGE_SIZE, TILE_SIZE, build_tiled_dataset, read_labels

SEED = 42
WORKSPACE, PROJECT, VERSION = "defect-detection-witqu", "fabric-defect-thesis", 1
NAMES = ["Hole", "Seam", "Stitch"]

RAW_DIR = Path("datasets") / f"{PROJECT}-{VERSION}"
TILE_DIR = Path("datasets") / f"{PROJECT}-{VERSION}-tiles"
RUNS = Path("runs").resolve()
ARTIFACTS = Path("artifacts")
FIG_DIR = ARTIFACTS / "figures"
WEIGHTS = Path("weights") / "fabric_defect_yolo11s.pt"
FIG_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "yolo11s.pt"  # COCO-pretrained: the finetuning start point and the zero-shot baseline
EPOCHS, PATIENCE, BATCH = 1, 20, 16
THREADS, WORKERS = 8, 4  # below the core count: sustained all-core load has crashed this laptop
IMGSZ = TILE_SIZE  # tiles are trained at their native 320 px, no upscaling
DEVICE = 0 if torch.cuda.is_available() else "cpu"
torch.set_num_threads(THREADS)
SCORE_FLOOR = 0.01  # detections below this confidence are discarded before scoring
print("Training on", torch.cuda.get_device_name(0) if DEVICE == 0 else "CPU")

ASSUMPTIONS = {
    "cost": {
        "tp": 0.0,  # PLACEHOLDER defect caught automatically (e.g. cut out / reworked)
        "fp": 1.0,  # PLACEHOLDER clean fabric pulled by mistake
        "fn": 5.0,  # PLACEHOLDER defect shipped to a customer
        "tn": 0.0,  # PLACEHOLDER clean fabric shipped
    },
    "review_cost": 0.1,  # PLACEHOLDER euros of a person's time per tile reviewed
    "base_rate": 0.05,  # PLACEHOLDER share of production tiles that contain a defect
    "tiles_per_day": 50_000,  # PLACEHOLDER tiles inspected per day
    "review_seconds": 5,  # PLACEHOLDER seconds for a person to check one tile
    "shift_hours": 8,  # PLACEHOLDER working hours per reviewer per day
}

rf = Roboflow(api_key=load_api_key())
dataset = rf.workspace(WORKSPACE).project(PROJECT).version(VERSION).download("yolov11", location=str(RAW_DIR))
print("Dataset in", RAW_DIR)

def source_of(path: Path) -> str:
    return path.name.split(".rf.")[0].rsplit("_", 1)[0]


split_rows, box_rows = [], []
for split in ["train", "valid", "test"]:
    images = sorted((RAW_DIR / split / "images").iterdir())
    split_rows.append({"split": split, "images": len(images), "unique_originals": len({source_of(p) for p in images})})
    for p in images:
        classes, boxes = read_labels(RAW_DIR / split / "labels" / f"{p.stem}.txt")
        for c, (x1, y1, x2, y2) in zip(classes, boxes):
            box_rows.append({"split": split, "class": NAMES[c], "area_pct": 100 * (x2 - x1) * (y2 - y1) / IMAGE_SIZE**2})

raw_splits = pd.DataFrame(split_rows).set_index("split")
raw_boxes = pd.DataFrame(box_rows)
raw_instances = raw_boxes.pivot_table(index="split", columns="class", values="area_pct", aggfunc="size").loc[raw_splits.index]
raw_sizes = raw_boxes.groupby(["split", "class"])["area_pct"].describe(percentiles=[0.1, 0.5, 0.9])[["10%", "50%", "90%"]].round(2)
print(raw_splits.join(raw_instances).to_string(), "\n")
print("Box area as % of image (10th / median / 90th percentile):")
print(raw_sizes.to_string())

if TILE_DIR.exists():
    shutil.rmtree(TILE_DIR)
manifest = build_tiled_dataset(RAW_DIR, TILE_DIR, NAMES)

status = manifest.groupby(["split", "status"]).size().unstack(fill_value=0)
kept = manifest[manifest["status"] == "kept"]
tiles = kept.groupby("split").agg(tiles=("file", "size"), defect_tiles=("defect", "sum"), **{n: (n, "sum") for n in NAMES})
tiles["clean_tiles"] = tiles["tiles"] - tiles["defect_tiles"]
tiles["defect_share"] = (tiles["defect_tiles"] / tiles["tiles"]).round(3)
tiles = tiles.loc[["train", "valid", "test"]]
print(status.to_string(), "\n")
print(tiles.to_string())

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained")
raw_instances.plot.bar(ax=axes[0], rot=0, title="Boxes per class (full images)")
tiles[["clean_tiles", "defect_tiles"]].plot.bar(ax=axes[1], rot=0, stacked=True, title="Tiles kept per split")
fig.savefig(FIG_DIR / "dataset.png", dpi=110)

run_dir = RUNS / "fabric"
best = run_dir / "weights" / "best.pt"
last = run_dir / "weights" / "last.pt"
if os.environ.get("REUSE_TRAINING") == "1" and best.exists():
    print("Reusing", best)
elif os.environ.get("RESUME_TRAINING") == "1" and last.exists():
    # picks an interrupted run back up from its last finished epoch, with its original settings
    subprocess.run([str(Path(sys.executable).with_name("yolo")), "detect", "train", "resume", f"model={last}"], check=True)
else:
    yolo_cli = Path(sys.executable).with_name("yolo")
    subprocess.run([
        str(yolo_cli), "detect", "train",
        f"model={BASE_MODEL}", f"data={(TILE_DIR / 'data.yaml').resolve()}",
        f"epochs={EPOCHS}", f"patience={PATIENCE}", f"imgsz={IMGSZ}", f"batch={BATCH}", f"workers={WORKERS}",
        f"device={DEVICE}", f"seed={SEED}", "deterministic=True",
        f"project={RUNS}", "name=fabric", "exist_ok=True",
    ], check=True)

WEIGHTS.parent.mkdir(exist_ok=True)
shutil.copy(best, WEIGHTS)
print("Saved", WEIGHTS)

history = pd.read_csv(run_dir / "results.csv")
history.columns = history.columns.str.strip()
print(f"{len(history)} epochs; best val mAP50-95 {history['metrics/mAP50-95(B)'].max():.3f} "
      f"at epoch {int(history.loc[history['metrics/mAP50-95(B)'].idxmax(), 'epoch'])}")

model = YOLO(str(WEIGHTS))
# workers=0: same Windows re-import problem as training, and the test set is small
val = model.val(data=str(TILE_DIR / "data.yaml"), split="test", imgsz=IMGSZ, batch=BATCH, device=DEVICE,
                workers=0, plots=True, project=str(RUNS), name="test_eval", exist_ok=True, verbose=False)

per_class = []
for i, c in enumerate(val.ap_class_index):
    p, r, ap50, ap = val.box.class_result(i)
    per_class.append({"class": NAMES[c], "test_boxes": int(tiles.loc["test", NAMES[c]]),
                      "precision": p, "recall": r, "AP50": ap50, "mAP50-95": ap})
per_class = pd.DataFrame(per_class).set_index("class")
per_class.loc["all"] = [per_class["test_boxes"].sum(), val.box.mp, val.box.mr, val.box.map50, val.box.map]
print(per_class.round(3).to_string())

confusion = val.confusion_matrix.matrix.astype(int)
labels = NAMES + ["background"]
print("\nConfusion matrix (rows predicted, columns true):")
print(pd.DataFrame(confusion, index=[f"pred {l}" for l in labels], columns=[f"true {l}" for l in labels]).to_string())
for f in ["confusion_matrix.png", "confusion_matrix_normalized.png", "BoxPR_curve.png"]:
    if (RUNS / "test_eval" / f).exists():
        shutil.copy(RUNS / "test_eval" / f, FIG_DIR / f"test_{f}")

def tile_paths(split: str) -> list[Path]:
    return sorted((TILE_DIR / split / "images").glob("*.jpg"))


def true_boxes(path: Path) -> np.ndarray:
    rows = [l.split() for l in path.read_text().splitlines() if l.strip()]
    xywh = np.array([r[1:] for r in rows], dtype=float).reshape(-1, 4) * TILE_SIZE
    return np.c_[xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2]


def detect(yolo: YOLO, paths: list[Path], imgsz: int) -> dict:
    """{tile name: (confidences, class ids, xyxy boxes in tile pixels)}."""
    out = {}
    for r in yolo.predict([str(p) for p in paths], imgsz=imgsz, conf=SCORE_FLOOR, device=DEVICE,
                          batch=BATCH, stream=True, verbose=False):
        b = r.boxes
        out[Path(r.path).stem] = (b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int), b.xyxy.cpu().numpy())
    return out


coco = YOLO(BASE_MODEL)
detections, score_rows = {"finetuned": {}, "coco": {}}, []
for split in ["valid", "test"]:
    paths = tile_paths(split)
    fine, zero = detect(model, paths, IMGSZ), detect(coco, paths, 640)
    detections["finetuned"] |= fine
    detections["coco"] |= zero
    for p in paths:
        conf, cls, _ = fine[p.stem]
        row = {"split": split, "file": p.stem, "score_finetuned": conf.max(initial=0.0),
               "score_coco": zero[p.stem][0].max(initial=0.0), "score_nothing": 0.0}
        row |= {f"score_{n}": conf[cls == i].max(initial=0.0) for i, n in enumerate(NAMES)}
        score_rows.append(row)

scores = pd.DataFrame(score_rows).merge(kept[["split", "file", "defect", *NAMES]], on=["split", "file"])
scores.to_csv(ARTIFACTS / "tile_scores.csv", index=False)
print(scores.groupby(["split", "defect"])[["score_finetuned", "score_coco"]].describe().T.round(3).to_string())

def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    tl, br = np.maximum(box[:2], boxes[:, :2]), np.minimum(box[2:], boxes[:, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=1)
    area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / (area(box) + area(boxes) - inter)


def agnostic_ap50(dets: dict, names: list[str]) -> float:
    """VOC-style all-point AP at IoU 0.5, every class merged into one."""
    hits, n_true = [], 0
    for name in names:
        truth = true_boxes(TILE_DIR / "test" / "labels" / f"{name}.txt")
        n_true += len(truth)
        used = np.zeros(len(truth), bool)
        conf, _, boxes = dets.get(name, (np.zeros(0), None, np.zeros((0, 4))))
        for j in np.argsort(-conf):
            hit = False
            if len(truth):
                iou = np.where(used, 0, box_iou(boxes[j], truth))
                k = iou.argmax()
                if iou[k] >= 0.5:
                    used[k], hit = True, True
            hits.append((conf[j], hit))
    if not hits or n_true == 0:
        return 0.0
    hits = np.array(sorted(hits, key=lambda h: -h[0]))
    tp = np.cumsum(hits[:, 1])
    recall, precision = tp / n_true, tp / np.arange(1, len(hits) + 1)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    return float(np.sum(np.diff(np.r_[0, recall]) * precision))


def roc_auc(score: np.ndarray, positive: np.ndarray) -> float:
    ranks = pd.Series(score).rank().to_numpy()
    n_pos, n_neg = positive.sum(), (~positive).sum()
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


test = scores[scores["split"] == "test"]
test_defect = test["defect"].to_numpy()
baselines = pd.DataFrame([
    {"model": "finetuned YOLO11s", "agnostic_AP50": agnostic_ap50(detections["finetuned"], test["file"]),
     "mAP50-95 (3 classes)": val.box.map, "tile_ROC_AUC": roc_auc(test["score_finetuned"].to_numpy(), test_defect)},
    {"model": "COCO YOLO11s zero-shot", "agnostic_AP50": agnostic_ap50(detections["coco"], test["file"]),
     "mAP50-95 (3 classes)": np.nan, "tile_ROC_AUC": roc_auc(test["score_coco"].to_numpy(), test_defect)},
    {"model": "predict nothing", "agnostic_AP50": 0.0, "mAP50-95 (3 classes)": 0.0, "tile_ROC_AUC": 0.5},
]).set_index("model")
print(baselines.round(3).to_string())

GRID_T = np.round(np.linspace(0, 1, 101), 2)
val_tiles = scores[scores["split"] == "valid"]
val_defect = val_tiles["defect"].to_numpy()

sweep = pd.DataFrame(threshold_sweep(val_tiles["score_finetuned"].to_numpy(), val_defect, ASSUMPTIONS, GRID_T))
sweep = sweep.add_prefix("valid_").rename(columns={"valid_threshold": "threshold"})
test_sweep = pd.DataFrame(threshold_sweep(test["score_finetuned"].to_numpy(), test_defect, ASSUMPTIONS, GRID_T))
sweep = sweep.join(test_sweep.drop(columns="threshold").add_prefix("test_"))
coco_sweep = threshold_sweep(test["score_coco"].to_numpy(), test_defect, ASSUMPTIONS, GRID_T)
sweep["test_coco_cost_per_tile"] = [r["cost_per_tile"] for r in coco_sweep]
sweep.to_csv(ARTIFACTS / "cost_sweep.csv", index=False)

THRESHOLD = float(sweep.loc[sweep["valid_cost_per_tile"].idxmin(), "threshold"])
single = outcome_rates(test["score_finetuned"].to_numpy(), test_defect, THRESHOLD, THRESHOLD)
print(f"Chosen threshold (min validation cost): {THRESHOLD:.2f}")
print(f"Test: recall {single['fail|defect']:.3f}, false alarms {single['fail|clean']:.3f}, "
      f"cost per 1,000 tiles EUR {1000 * expected_cost(single, ASSUMPTIONS):,.2f}")

fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
ax.plot(sweep["threshold"], 1000 * sweep["valid_cost_per_tile"], label="finetuned (validation)")
ax.plot(sweep["threshold"], 1000 * sweep["test_cost_per_tile"], label="finetuned (test)")
ax.plot(sweep["threshold"], 1000 * sweep["test_coco_cost_per_tile"], label="COCO zero-shot (test)", linestyle=":")
for name, c in alternatives(ASSUMPTIONS).items():
    ax.axhline(1000 * c, color="grey", linestyle="--", linewidth=1)
    ax.annotate(name, (1, 1000 * c), xytext=(-4, 3), textcoords="offset points", ha="right", fontsize=8, color="grey")
ax.axvline(THRESHOLD, color="black", linewidth=1)
ax.set(xlabel="confidence threshold", ylabel="expected EUR per 1,000 tiles", title="Cost against threshold")
ax.legend()
fig.savefig(FIG_DIR / "cost_vs_threshold.png", dpi=110)

band = best_band(val_tiles["score_finetuned"].to_numpy(), val_defect, ASSUMPTIONS, GRID_T)
T_LOW, T_HIGH = band["t_low"], band["t_high"]
band_rates = outcome_rates(test["score_finetuned"].to_numpy(), test_defect, T_LOW, T_HIGH)
share = review_share(band_rates, ASSUMPTIONS["base_rate"])

designs = {name: c for name, c in alternatives(ASSUMPTIONS).items()}
designs[f"model alone (threshold {THRESHOLD:.2f})"] = expected_cost(single, ASSUMPTIONS)
designs[f"model + review ({T_LOW:.2f}-{T_HIGH:.2f})"] = expected_cost(band_rates, ASSUMPTIONS)
review_shares = {name: 0.0 for name in designs}
review_shares["person checks every tile"] = 1.0
review_shares[f"model + review ({T_LOW:.2f}-{T_HIGH:.2f})"] = share
comparison = pd.DataFrame({
    "EUR per 1,000 tiles": {k: 1000 * v for k, v in designs.items()},
    "EUR per day": {k: v * ASSUMPTIONS["tiles_per_day"] for k, v in designs.items()},
    "review share": review_shares,
    "reviewers needed": {k: reviewers_needed(s, ASSUMPTIONS) for k, s in review_shares.items()},
})
print(f"Band: pass < {T_LOW:.2f} <= review < {T_HIGH:.2f} <= fail")
print(pd.Series(per_thousand(band_rates, ASSUMPTIONS["base_rate"])).round(1).to_string(), "\n")
print(comparison.round(2).to_string())

metrics = {
    "dataset": {
        "workspace": WORKSPACE, "project": PROJECT, "version": VERSION, "classes": NAMES,
        "raw_splits": raw_splits.join(raw_instances).reset_index().to_dict("records"),
        "box_area_pct": raw_sizes.reset_index().to_dict("records"),
        "tiles": tiles.reset_index().to_dict("records"),
        "tile_status": status.reset_index().to_dict("records"),
        "grid": GRID, "tile_size": TILE_SIZE,
    },
    "training": {"base_model": BASE_MODEL, "epochs_run": len(history), "max_epochs": EPOCHS, "patience": PATIENCE,
                 "imgsz": IMGSZ, "batch": BATCH, "seed": SEED,
                 "device": torch.cuda.get_device_name(0) if DEVICE == 0 else "cpu"},
    "per_class": per_class.reset_index().to_dict("records"),
    "confusion_matrix": {"labels": labels, "matrix": confusion.tolist(), "orientation": "rows predicted, columns true"},
    "baselines": baselines.reset_index().replace({np.nan: None}).to_dict("records"),
    "assumptions": ASSUMPTIONS,
    "operating_point": {"threshold": THRESHOLD, "t_low": T_LOW, "t_high": T_HIGH, "chosen_on": "valid"},
    "comparison": comparison.reset_index(names="design").to_dict("records"),
}
(ARTIFACTS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
print("Wrote", sorted(p.name for p in ARTIFACTS.iterdir()))
