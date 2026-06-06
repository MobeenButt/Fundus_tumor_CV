"""
Auto-Annotation Script — Fundus Tumor Dataset
==============================================
Generates YOLO-format bounding box annotations for ALL images
using GradCAM from the trained EfficientNet-B0 classifier.

Supports TWO modes:
  MODE 1 (local)  — data already in data/Training & data/Testing
  MODE 2 (kaggle) — downloads dataset directly from Kaggle

Output structure:
  annotations/
    images/train/   ← training images
    images/val/     ← test/val images
    labels/train/   ← YOLO .txt files  (class_id cx cy w h)
    labels/val/     ← YOLO .txt files
    classes.txt
    dataset.yaml    ← ready for YOLOv8 training

Run locally  : python generate_annotations.py
Run on Colab : python generate_annotations.py --kaggle
               (needs kaggle.json in same folder)
"""

import argparse
import os
import sys
import shutil
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import cv2
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENT PARSING
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--kaggle', action='store_true',
                    help='Download dataset from Kaggle instead of using local data')
parser.add_argument('--kaggle_json', type=str, default='kaggle.json',
                    help='Path to kaggle.json (default: kaggle.json in current folder)')
args = parser.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# KAGGLE DOWNLOAD (if --kaggle flag used)
# ─────────────────────────────────────────────────────────────────────────────
if args.kaggle:
    print("="*60)
    print("  MODE: Kaggle Download")
    print("="*60)

    kaggle_json = Path(args.kaggle_json)
    if not kaggle_json.exists():
        print(f"❌ kaggle.json not found at: {kaggle_json}")
        print("   Download it from: kaggle.com → Account → API → Create New Token")
        sys.exit(1)

    # Setup credentials
    kaggle_dir = Path.home() / '.config' / 'kaggle'
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(kaggle_json, kaggle_dir / 'kaggle.json')
    os.chmod(kaggle_dir / 'kaggle.json', 0o600)
    print("✓ Kaggle credentials configured")

    # Install kaggle if needed
    try:
        import kaggle
    except ImportError:
        print("Installing kaggle package...")
        os.system("pip install -q kaggle")
        import kaggle

    # Download dataset
    DATASET_SLUG = 'nikitamanaenkov/ultra-wide-fundus-images-for-tumor-diagnosis'
    DOWNLOAD_DIR = Path('kaggle_data')
    DOWNLOAD_DIR.mkdir(exist_ok=True)

    print(f"⬇️  Downloading: {DATASET_SLUG}")
    os.system(f'kaggle datasets download -d {DATASET_SLUG} --unzip -p {DOWNLOAD_DIR}')
    print("✓ Dataset downloaded")

    # Auto-detect Training/Testing folders
    def find_split(root, name):
        matches = list(root.rglob(name))
        return matches[0] if matches else None

    TRAIN_DIR = find_split(DOWNLOAD_DIR, 'Training') or DOWNLOAD_DIR
    TEST_DIR  = find_split(DOWNLOAD_DIR, 'Testing')  or DOWNLOAD_DIR
    print(f"✓ TRAIN_DIR: {TRAIN_DIR}")
    print(f"✓ TEST_DIR : {TEST_DIR}")

else:
    print("="*60)
    print("  MODE: Local Data")
    print("="*60)
    TRAIN_DIR = Path('data/Training')
    TEST_DIR  = Path('data/Testing')

    if not TRAIN_DIR.exists():
        print(f"❌ {TRAIN_DIR} not found.")
        print("   Use --kaggle flag to download, or make sure data/ folder exists.")
        sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE     = 224
MODEL_PATH   = Path('model/best_model.pth')
CLASSES_PATH = Path('model/classes.json')
OUT_DIR      = Path('annotations')
GRADCAM_THR  = 0.40

print(f"\nDevice : {DEVICE}")
print(f"Output : {OUT_DIR}")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────
if not MODEL_PATH.exists():
    print(f"❌ Model not found: {MODEL_PATH}")
    print("   Make sure model/best_model.pth exists")
    sys.exit(1)

with open(CLASSES_PATH) as f:
    CLASS_NAMES = json.load(f)
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

def build_model(n):
    m = models.efficientnet_b0(weights=None)
    m.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(m.classifier[1].in_features, n)
    )
    return m

model = build_model(NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE).eval()
print(f"✓ Model loaded — {NUM_CLASSES} classes: {CLASS_NAMES}")

# ─────────────────────────────────────────────────────────────────────────────
# GRADCAM
# ─────────────────────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.grads = self.acts = None
        layer = model.features[-1]
        layer.register_forward_hook(
            lambda m, i, o: setattr(self, 'acts', o.detach()))
        layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, 'grads', go[0].detach()))

    def generate(self, tensor):
        self.model.zero_grad()
        out = self.model(tensor)
        cls_idx = out.argmax(1).item()
        out[0, cls_idx].backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * self.acts).sum(1)).squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, cls_idx

gradcam = GradCAM(model)

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────────────────────────────────────
# BBOX EXTRACTION — YOLO FORMAT
# ─────────────────────────────────────────────────────────────────────────────
def cam_to_yolo_bbox(cam, img_w, img_h, threshold=GRADCAM_THR):
    """Returns YOLO: cx, cy, w, h — all normalized 0 to 1."""
    cam_r = cv2.resize(cam, (img_w, img_h))
    binary = (cam_r >= threshold).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 0.5, 0.5, 0.4, 0.4  # fallback center box

    pts = np.concatenate(contours)
    x, y, w, h = cv2.boundingRect(pts)

    # 10% padding
    pad_x = int(w * 0.10)
    pad_y = int(h * 0.10)
    x = max(0, x - pad_x)
    y = max(0, y - pad_y)
    w = min(img_w - x, w + 2 * pad_x)
    h = min(img_h - y, h + 2 * pad_y)

    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return round(cx, 6), round(cy, 6), round(nw, 6), round(nh, 6)

# ─────────────────────────────────────────────────────────────────────────────
# ANNOTATE A SPLIT
# ─────────────────────────────────────────────────────────────────────────────
def annotate_split(split_dir, split_name):
    img_out = OUT_DIR / 'images' / split_name
    lbl_out = OUT_DIR / 'labels' / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    total, skipped = 0, 0
    class_counts = {c: 0 for c in CLASS_NAMES}

    for class_name in CLASS_NAMES:
        class_dir = split_dir / class_name
        if not class_dir.exists():
            print(f"  ⚠️  Skipping (not found): {class_dir}")
            continue

        class_id = CLASS_TO_ID[class_name]
        img_files = sorted(list(class_dir.glob('*.jpg')) +
                           list(class_dir.glob('*.png')) +
                           list(class_dir.glob('*.jpeg')))

        for img_path in tqdm(img_files, desc=f"  [{split_name}] {class_name[:28]:<28}"):
            try:
                pil = Image.open(img_path).convert('RGB')
                orig_w, orig_h = pil.size

                tensor = transform(pil).unsqueeze(0).to(DEVICE)
                cam, pred_idx = gradcam.generate(tensor)

                cx, cy, nw, nh = cam_to_yolo_bbox(cam, orig_w, orig_h)

                # Copy image
                shutil.copy2(img_path, img_out / img_path.name)

                # Write YOLO label
                with open(lbl_out / (img_path.stem + '.txt'), 'w') as f:
                    f.write(f"{class_id} {cx} {cy} {nw} {nh}\n")

                class_counts[class_name] += 1
                total += 1

            except Exception as e:
                skipped += 1

    return total, skipped, class_counts

# ─────────────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  ANNOTATING TRAINING SET")
print("="*60)
train_total, train_skip, train_counts = annotate_split(TRAIN_DIR, 'train')

print("\n" + "="*60)
print("  ANNOTATING TEST / VAL SET")
print("="*60)
val_total, val_skip, val_counts = annotate_split(TEST_DIR, 'val')

# ─────────────────────────────────────────────────────────────────────────────
# SAVE classes.txt + dataset.yaml
# ─────────────────────────────────────────────────────────────────────────────
with open(OUT_DIR / 'classes.txt', 'w') as f:
    for name in CLASS_NAMES:
        f.write(name + '\n')

yaml_content = f"""# YOLOv8 Dataset Config — Fundus Tumor Detection
# Generated by generate_annotations.py

path: {OUT_DIR.resolve().as_posix()}
train: images/train
val:   images/val

nc: {NUM_CLASSES}
names: {CLASS_NAMES}
"""
with open(OUT_DIR / 'dataset.yaml', 'w') as f:
    f.write(yaml_content)

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  ANNOTATION COMPLETE")
print("="*60)
print(f"\n  Training : {train_total} annotated  |  {train_skip} skipped")
for cls, cnt in train_counts.items():
    if cnt > 0:
        print(f"    {cls:<42} {cnt:>5} images")

print(f"\n  Val/Test : {val_total} annotated  |  {val_skip} skipped")
for cls, cnt in val_counts.items():
    if cnt > 0:
        print(f"    {cls:<42} {cnt:>5} images")

print(f"\n  Total    : {train_total + val_total} images annotated")
print(f"\n  Output:")
print(f"    annotations/images/train/   ({train_total} images)")
print(f"    annotations/images/val/     ({val_total} images)")
print(f"    annotations/labels/train/   ({train_total} .txt files)")
print(f"    annotations/labels/val/     ({val_total} .txt files)")
print(f"    annotations/classes.txt")
print(f"    annotations/dataset.yaml    ← use for YOLOv8")
print("="*60)
print("\n✅ Done!")
if args.kaggle:
    print("   Upload the 'annotations/' folder to Colab/Drive for YOLOv8 training.")
