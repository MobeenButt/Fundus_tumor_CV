"""
YOLOv8 Fundus Tumor Detection — Inference & Results
====================================================
Loads best.pt, runs detection on test images,
saves annotated images and generates metrics.txt.
"""

from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH   = Path(__file__).parent / "best.pt"
TEST_DIR     = Path(__file__).parent.parent / "data" / "Testing"
OUT_DIR      = Path(__file__).parent / "detected_images"
CLASSES_FILE = Path(__file__).parent.parent / "model" / "classes.json"

CONF_THRESH  = 0.25
IMG_SIZE     = 640
DEVICE       = "cpu"

CLASS_NAMES = [
    "Choroidal Hemangioma (CH)", "Choroidal Osteoma (CO)", "Normal",
    "Retinal Capillary Hemangioma (RCH)", "Retinoblastoma (RB)", "Uveal Melanoma (UM)"
]

CLASS_SHORT = {
    "Choroidal Hemangioma (CH)": "CH",
    "Choroidal Osteoma (CO)": "CO",
    "Normal": "Normal",
    "Retinal Capillary Hemangioma (RCH)": "RCH",
    "Retinoblastoma (RB)": "RB",
    "Uveal Melanoma (UM)": "UM",
}

CLASS_COLORS = {
    "CH": "#FF6B6B", "CO": "#4ECDC4", "Normal": "#95E1D3",
    "RCH": "#FFD93D", "RB": "#FF006E", "UM": "#6A4C93",
}

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load model ────────────────────────────────────────────────────────────────
print(f"Loading model: {MODEL_PATH}")
model = YOLO(str(MODEL_PATH))
print("Model loaded successfully\n")

# ── Collect test images ───────────────────────────────────────────────────────
image_paths = []
for cls_dir in sorted(TEST_DIR.iterdir()):
    if cls_dir.is_dir():
        for ext in ("*.jpg", "*.png", "*.jpeg"):
            image_paths.extend(cls_dir.glob(ext))

image_paths = sorted(image_paths)
print(f"Found {len(image_paths)} test images\n")

# ── Run detection ─────────────────────────────────────────────────────────────
results_data = []
for img_path in image_paths:
    gt_class = img_path.parent.name
    print(f"  Detecting: {img_path.name}  (GT: {CLASS_SHORT.get(gt_class, gt_class)})")

    results = model.predict(
        str(img_path),
        conf=CONF_THRESH,
        imgsz=IMG_SIZE,
        device=DEVICE,
        verbose=False,
    )
    res = results[0]

    orig = cv2.imread(str(img_path))
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    img_h, img_w = orig.shape[:2]

    detections = []
    if res.boxes is not None:
        for box in res.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "Unknown"
            short = CLASS_SHORT.get(cls_name, "?")
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": round(conf, 4),
                "class_id": cls_id,
                "class_name": cls_name,
                "short": short,
            })

    is_correct = any(d["class_name"] == gt_class for d in detections)
    results_data.append({
        "filename": img_path.name,
        "gt_class": gt_class,
        "gt_short": CLASS_SHORT.get(gt_class, gt_class),
        "detections": detections,
        "correct": is_correct,
        "num_detections": len(detections),
        "img_w": img_w,
        "img_h": img_h,
    })

# ── Generate annotated images ─────────────────────────────────────────────────
print("\nSaving annotated images...")

for rd in results_data:
    img_path = next(
        p for cls_dir in TEST_DIR.iterdir()
        if cls_dir.is_dir()
        for p in cls_dir.glob(rd["filename"])
    )
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(img)

    for det in rd["detections"]:
        x1, y1, x2, y2 = det["bbox"]
        color = CLASS_COLORS.get(det["short"], "#FF0000")
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=3, edgecolor=color, facecolor="none",
        )
        ax.add_patch(rect)
        label = f"{det['short']} {det['confidence']:.2f}"
        ax.text(
            x1, max(y1 - 8, 10), label, color="white", fontsize=10,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
        )

    status = "CORRECT" if rd["correct"] else "MISS"
    status_color = "green" if rd["correct"] else "red"
    ax.set_title(
        f"GT: {rd['gt_short']}  |  Detected: {rd['num_detections']}  [{status}]",
        fontsize=13, fontweight="bold", color=status_color,
    )
    ax.axis("off")
    plt.tight_layout()
    out_path = OUT_DIR / f"{Path(rd['filename']).stem}_detected.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

print(f"  Saved {len(results_data)} annotated images to {OUT_DIR}/\n")

# ── Compute metrics ───────────────────────────────────────────────────────────
total = len(results_data)
correct = sum(1 for r in results_data if r["correct"])
acc = correct / total * 100 if total > 0 else 0

per_class = {}
for rd in results_data:
    cls = rd["gt_class"]
    if cls not in per_class:
        per_class[cls] = {"total": 0, "correct": 0}
    per_class[cls]["total"] += 1
    if rd["correct"]:
        per_class[cls]["correct"] += 1

# ── Save results ──────────────────────────────────────────────────────────────
results_json = {
    "total_images": total,
    "correct_detections": correct,
    "accuracy": round(acc, 2),
    "per_class": {
        cls: {
            "total": v["total"],
            "correct": v["correct"],
            "accuracy": round(v["correct"] / v["total"] * 100, 2),
        }
        for cls, v in per_class.items()
    },
    "details": [
        {
            "filename": r["filename"],
            "gt_class": r["gt_class"],
            "correct": r["correct"],
            "detections": [
                {
                    "class": d["class_name"],
                    "confidence": d["confidence"],
                    "bbox": d["bbox"],
                }
                for d in r["detections"]
            ],
        }
        for r in results_data
    ],
}

json_path = OUT_DIR / "results.json"
with open(json_path, "w") as f:
    json.dump(results_json, f, indent=2)
print(f"Saved: {json_path}")

# ── Generate metrics.txt ──────────────────────────────────────────────────────
lines = []
lines.append("=" * 80)
lines.append("  YOLOv8 Detection Results — Fundus Tumor Dataset")
lines.append("=" * 80)
lines.append("")
lines.append(f"  Model        : YOLOv8n")
lines.append(f"  Weights      : {MODEL_PATH.name}")
lines.append(f"  Confidence   : {CONF_THRESH}")
lines.append(f"  Image Size   : {IMG_SIZE}x{IMG_SIZE}")
lines.append(f"  Test Images  : {total}")
lines.append("")
lines.append("-" * 80)
lines.append("  Overall Detection Accuracy")
lines.append("-" * 80)
lines.append(f"  Correct      : {correct}/{total}")
lines.append(f"  Accuracy     : {acc:.2f}%")
lines.append("")
lines.append("-" * 80)
lines.append("  Per-Class Detection Results")
lines.append("-" * 80)
lines.append(f"  {'Class':<42} {'Total':>6} {'Correct':>8} {'Accuracy':>10}")
lines.append(f"  {'─'*42} {'─'*6} {'─'*8} {'─'*10}")
for cls_name in CLASS_NAMES:
    v = per_class.get(cls_name, {"total": 0, "correct": 0})
    pct = round(v["correct"] / v["total"] * 100, 2) if v["total"] > 0 else 0
    lines.append(f"  {cls_name:<42} {v['total']:>6} {v['correct']:>8} {pct:>9.2f}%")
lines.append("")
lines.append("-" * 80)
lines.append("  Per-Image Detection Summary")
lines.append("-" * 80)
lines.append(f"  {'Image':<30} {'Ground Truth':<35} {'Detections':>10} {'Status':>10}")
lines.append(f"  {'─'*30} {'─'*35} {'─'*10} {'─'*10}")
for rd in results_data:
    det_classes = ", ".join(d["short"] for d in rd["detections"]) or "none"
    status = "OK" if rd["correct"] else "MISS"
    lines.append(
        f"  {rd['filename']:<30} {rd['gt_short']:<35} {det_classes:>10} {status:>10}"
    )
lines.append("")
lines.append("=" * 80)
lines.append(f"  FINAL ACCURACY: {acc:.2f}%  ({correct}/{total} correct)")
lines.append("=" * 80)

metrics_txt = OUT_DIR / "metrices.txt"
with open(metrics_txt, "w") as f:
    f.write("\n".join(lines))
print(f"Saved: {metrics_txt}")

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print(f"  DETECTION COMPLETE")
print(f"  Accuracy: {acc:.2f}%  ({correct}/{total})")
print(f"  Results:  {OUT_DIR}/")
print(f"  ├── metrics.txt")
print(f"  ├── results.json")
print(f"  └── *_detected.png  ({len(results_data)} files)")
print("=" * 80)
