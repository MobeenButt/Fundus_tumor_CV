import os
import json
import io
import base64
import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision import models
import numpy as np
import cv2

from flask import (
    Flask, render_template, send_file, request, jsonify, url_for, Response,
)
from PIL import Image, ImageDraw

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR         = Path(__file__).resolve().parent
ANNOTATIONS_FILE = BASE_DIR / "annotations_flask.json"
VISUAL_DIR       = BASE_DIR / "annotated_images"
MODEL_DIR        = BASE_DIR.parent / "model"
MODEL_PATH       = MODEL_DIR / "best_model.pth"
CLASSES_PATH     = MODEL_DIR / "classes.json"
UNET_PATH        = MODEL_DIR / "segmentation_results" / "best_unet.pth"

# Check multiple locations for U-Net
UNET_CANDIDATES = [
    BASE_DIR.parent / "Week4_Segmentation_Results" / "best_unet.pth",
    MODEL_DIR / "segmentation_results" / "best_unet.pth",
    BASE_DIR / "best_unet.pth",
]
UNET_PATH = next((p for p in UNET_CANDIDATES if p.exists()), None)

CLASSES = [
    "Choroidal Hemangioma (CH)",
    "Choroidal Osteoma (CO)",
    "Normal",
    "Retinal Capillary Hemangioma (RCH)",
    "Retinoblastoma (RB)",
    "Uveal Melanoma (UM)",
]
CLASS_COLORS = {
    "Choroidal Hemangioma (CH)": "#FF6B6B",
    "Choroidal Osteoma (CO)": "#4ECDC4",
    "Normal": "#95E1D3",
    "Retinal Capillary Hemangioma (RCH)": "#FFD93D",
    "Retinoblastoma (RB)": "#FF006E",
    "Uveal Melanoma (UM)": "#6A4C93",
}
CLASS_COLORS_RGB = {
    "Choroidal Hemangioma (CH)": (255, 107, 107),
    "Choroidal Osteoma (CO)": (78, 205, 196),
    "Normal": (149, 225, 211),
    "Retinal Capillary Hemangioma (RCH)": (255, 217, 61),
    "Retinoblastoma (RB)": (255, 0, 110),
    "Uveal Melanoma (UM)": (106, 76, 147),
}
FOLDER_TO_CLASS = {
    "choroidal hemangioma (ch)": "Choroidal Hemangioma (CH)",
    "choroidal osteoma (co)": "Choroidal Osteoma (CO)",
    "normal": "Normal",
    "retinal capillary hemangioma (rch)": "Retinal Capillary Hemangioma (RCH)",
    "retinoblastoma (rb)": "Retinoblastoma (RB)",
    "uveal melanoma (um)": "Uveal Melanoma (UM)",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load AI models ────────────────────────────────────────────────────────────
clf_model = None
gradcam   = None
unet      = None

def _build_classifier(n):
    m = models.efficientnet_b0(weights=None)
    m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.classifier[1].in_features, n))
    return m

class _GradCAM:
    def __init__(self, model):
        self.model = model
        self.grads = self.acts = None
        layer = model.features[-1]
        layer.register_forward_hook(lambda m,i,o: setattr(self,'acts',o.detach()))
        layer.register_full_backward_hook(lambda m,gi,go: setattr(self,'grads',go[0].detach()))
    def generate(self, tensor):
        self.model.zero_grad()
        out = self.model(tensor)
        probs = torch.softmax(out, dim=1).squeeze().detach().cpu().numpy()
        cls_idx = int(out.argmax(1).item())
        out[0, cls_idx].backward()
        w = self.grads.mean(dim=(2,3), keepdim=True)
        cam = torch.relu((w * self.acts).sum(1)).squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, cls_idx, probs

class _DoubleConv(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ic,oc,3,padding=1,bias=False), nn.BatchNorm2d(oc), nn.ReLU(True),
            nn.Conv2d(oc,oc,3,padding=1,bias=False), nn.BatchNorm2d(oc), nn.ReLU(True))
    def forward(self, x): return self.net(x)

class _UNet(nn.Module):
    def __init__(self, ic=3, oc=1, feats=[64,128,256,512]):
        super().__init__()
        self.downs = nn.ModuleList(); self.ups = nn.ModuleList()
        self.pool  = nn.MaxPool2d(2)
        ch = ic
        for f in feats: self.downs.append(_DoubleConv(ch,f)); ch=f
        self.bot = _DoubleConv(feats[-1], feats[-1]*2)
        for f in reversed(feats):
            self.ups.append(nn.ConvTranspose2d(f*2,f,2,2))
            self.ups.append(_DoubleConv(f*2,f))
        self.final = nn.Conv2d(feats[0],oc,1)
    def forward(self, x):
        skips = []
        for d in self.downs: x=d(x); skips.append(x); x=self.pool(x)
        x = self.bot(x); skips = skips[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x); s = skips[i//2]
            if x.shape != s.shape: x = TF.resize(x, s.shape[2:])
            x = torch.cat([s,x],1); x = self.ups[i+1](x)
        return torch.sigmoid(self.final(x))

if MODEL_PATH.exists():
    clf_model = _build_classifier(len(CLASSES))
    clf_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    clf_model.to(DEVICE).eval()
    gradcam = _GradCAM(clf_model)
    print(f"✅ Classifier loaded on {DEVICE}")
else:
    print(f"⚠️  Classifier not found: {MODEL_PATH}")

if UNET_PATH and UNET_PATH.exists():
    unet = _UNet().to(DEVICE)
    unet.load_state_dict(torch.load(UNET_PATH, map_location=DEVICE))
    unet.eval()
    print(f"✅ U-Net loaded from {UNET_PATH.name} — IoU=67.13% Dice=79.71%")
else:
    print(f"⚠️  U-Net not found — run Week4 notebook first")

# ── YOLOv8 Detection ──────────────────────────────────────────────────────────
yolo_model = None
YOLO_CANDIDATES = [
    BASE_DIR.parent / "Week3_YOLO_Results" / "best.pt",
    BASE_DIR.parent / "model" / "best.pt",
    BASE_DIR / "best.pt",
    Path("d:/6 Sem/computer vision/fundus_tumor/Week3_YOLO_Results/best.pt"),
]
YOLO_PATH = next((p for p in YOLO_CANDIDATES if p.exists()), None)
print(f"YOLO search paths:")
for p in YOLO_CANDIDATES:
    print(f"  {'✅' if p.exists() else '❌'} {p}")

if YOLO_PATH:
    try:
        from ultralytics import YOLO as _YOLO
        yolo_model = _YOLO(str(YOLO_PATH))
        print(f"✅ YOLOv8 loaded: {YOLO_PATH.name} — mAP@0.5=98.55%")
    except ImportError:
        print("⚠️  ultralytics not installed — pip install ultralytics")
    except Exception as e:
        print(f"⚠️  YOLOv8 load failed: {e}")
else:
    print(f"⚠️  YOLOv8 best.pt not found — using GradCAM fallback")

clf_tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
seg_tf = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

app = Flask(__name__)
VISUAL_DIR.mkdir(exist_ok=True)




# ── Image scanner ─────────────────────────────────────────────────────────────
def scan_images(data_dir=None):
    if data_dir is None:
        # Try multiple locations
        candidates = [
            BASE_DIR.parent / "data",                    # fundus_tumor/data/
            BASE_DIR.parent / "data" / "data",           # nested data/data/
            BASE_DIR.parent,                              # fundus_tumor/
        ]
        data_dir = next((p for p in candidates
                         if (p / "Training").exists() or (p / "Choroidal Hemangioma (CH)").exists()),
                        BASE_DIR.parent)
    images = []
    data_dir = Path(data_dir)

    # Try Training/Testing split
    for split in ("Training", "Testing"):
        for folder_name, class_name in FOLDER_TO_CLASS.items():
            # Try exact case and title case
            for fn in [folder_name, folder_name.title(), class_name]:
                class_path = data_dir / split / fn
                if class_path.exists():
                    for f in sorted(class_path.iterdir()):
                        if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                            images.append({
                                "path": str(f.resolve()),
                                "class": class_name,
                                "split": split,
                                "filename": f.name,
                            })
                    break  # found this class, move on

    # Also include uploaded_images folder
    upload_dir = BASE_DIR / "uploaded_images"
    if upload_dir.exists():
        for f in sorted(upload_dir.iterdir()):
            if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                images.append({
                    "path": str(f.resolve()),
                    "class": "Normal",
                    "split": "Training",
                    "filename": f.name,
                })

    print(f"✅ Scanned {len(images)} images from {data_dir}")
    return images


IMAGE_LIST = scan_images()


# ── Annotation persistence ────────────────────────────────────────────────────
def load_annotations():
    if ANNOTATIONS_FILE.exists():
        try:
            with open(ANNOTATIONS_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_annotations(data):
    with open(ANNOTATIONS_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ── Annotated image generator ────────────────────────────────────────────────
def draw_annotations(img_path, boxes, labels, box_types=None, segmentations=None):
    pil_img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil_img, 'RGBA')
    
    # Draw segmentations first (as background)
    if segmentations:
        for seg_data in segmentations:
            if seg_data.get('points'):
                points = [tuple(p) for p in seg_data['points']]
                color = seg_data.get('color', '#FF0000')
                # Convert hex to RGB
                color_rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
                alpha_color = color_rgb + (80,)  # Semi-transparent
                if len(points) > 2:
                    draw.polygon(points, fill=alpha_color, outline=color_rgb + (255,))
    
    # Draw bounding boxes
    for i, (box, label) in enumerate(zip(boxes, labels)):
        box_type = box_types[i] if box_types and i < len(box_types) else "rectangle"
        color = CLASS_COLORS_RGB.get(label, (255, 0, 0))
        x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        
        if box_type == "rectangle":
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        elif box_type == "circle":
            # Draw circle from bounding box
            draw.ellipse([x1, y1, x2, y2], outline=color, width=4)
        elif box_type == "polygon":
            # If polygon data exists, use it
            if 'points' in box and box['points']:
                points = [tuple(p) for p in box['points']]
                draw.polygon(points, outline=color, width=4)
            else:
                # Fallback to rectangle
                draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        
        # Label badge
        font_size = max(12, (x2 - x1) // 15)
        try:
            from PIL import ImageFont
            font = ImageFont.truetype("arial.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pad = 4
        ly = max(0, y1 - th - pad * 2)
        draw.rectangle([x1, ly, x1 + tw + pad * 2, ly + th + pad * 2],
                       fill=color)
        tc = (0, 0, 0) if label == "No Tumor" else (255, 255, 255)
        draw.text((x1 + pad, ly + pad), label, fill=tc, font=font)
    
    return pil_img


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    global IMAGE_LIST
    total = len(IMAGE_LIST)
    return render_template("index.html", total=total, classes=CLASSES,
                           class_colors=CLASS_COLORS)


@app.route("/api/upload-images", methods=["POST"])
def api_upload_images():
    """Handle image uploads."""
    global IMAGE_LIST
    
    if 'files' not in request.files:
        return jsonify({"error": "No files provided"}), 400
    
    files = request.files.getlist('files')
    if not files:
        return jsonify({"error": "No files selected"}), 400
    
    # Create a directory for uploaded images
    upload_dir = BASE_DIR / "uploaded_images"
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    uploaded_count = 0
    for file in files:
        if file and file.filename:
            # Check if file is an image
            if file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                # Save file
                filepath = upload_dir / file.filename
                file.save(str(filepath))
                
                # Add to IMAGE_LIST
                IMAGE_LIST.append({
                    "path": str(filepath.resolve()),
                    "class": "No Tumor",
                    "split": "Training",
                    "filename": file.filename,
                })
                uploaded_count += 1
    
    if uploaded_count == 0:
        return jsonify({"error": "No valid image files uploaded"}), 400
    
    return jsonify({
        "success": True, 
        "uploaded": uploaded_count, 
        "total": len(IMAGE_LIST)
    })


@app.route("/api/images")
def api_images():
    """Return image list with annotation status."""
    annotations = load_annotations()
    result = []
    for idx, img in enumerate(IMAGE_LIST):
        str_idx = str(idx)
        ann = annotations.get(str_idx, {})
        result.append({
            "idx": idx,
            "filename": img["filename"],
            "class": img["class"],
            "split": img["split"],
            "annotated": bool(ann),
            "boxes": ann.get("boxes", []),
            "labels": ann.get("labels", []),
        })
    return jsonify(result)


@app.route("/api/image/<int:idx>")
def api_image(idx):
    """Return current image info and binary."""
    if idx < 0 or idx >= len(IMAGE_LIST):
        return jsonify({"error": "Index out of range"}), 404
    img = IMAGE_LIST[idx]
    str_idx = str(idx)
    annotations = load_annotations()
    ann = annotations.get(str_idx, {})
    return jsonify({
        "idx": idx,
        "filename": img["filename"],
        "class": img["class"],
        "split": img["split"],
        "path": img["path"],
        "annotated": bool(ann),
        "boxes": ann.get("boxes", []),
        "labels": ann.get("labels", []),
        "box_types": ann.get("box_types", []),
        "segmentations": ann.get("segmentations", []),
    })


@app.route("/api/image_file/<int:idx>")
def api_image_file(idx):
    """Serve the raw image file (using manual read to avoid send_file path issues)."""
    if idx < 0 or idx >= len(IMAGE_LIST):
        return "Not found", 404
    path = IMAGE_LIST[idx]["path"]
    ext = os.path.splitext(path)[1].lower()
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(ext, "image/jpeg")
    try:
        with open(path, "rb") as f:
            data = f.read()
        return Response(data, mimetype=mime)
    except Exception as e:
        return str(e), 500


@app.route("/api/save/<int:idx>", methods=["POST"])
def api_save(idx):
    """Save annotations for an image."""
    if idx < 0 or idx >= len(IMAGE_LIST):
        return jsonify({"error": "Invalid index"}), 400
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data"}), 400
    
    boxes = data.get("boxes", [])
    labels = data.get("labels", [])
    segmentations = data.get("segmentations", [])  # New: segmentation masks
    box_types = data.get("box_types", [])  # New: type of box (rectangle, circle, polygon)
    
    # Validate
    if len(boxes) != len(labels):
        return jsonify({"error": "boxes/labels length mismatch"}), 400
    
    for box in boxes:
        for k in ("x1", "y1", "x2", "y2"):
            if k not in box:
                return jsonify({"error": f"Missing key {k} in box"}), 400

    annotations = load_annotations()
    str_idx = str(idx)
    img = IMAGE_LIST[idx]
    annotations[str_idx] = {
        "idx": idx,
        "filename": img["filename"],
        "class": img["class"],
        "split": img["split"],
        "path": img["path"],
        "boxes": boxes,
        "labels": labels,
        "box_types": box_types if box_types else ["rectangle"] * len(boxes),
        "segmentations": segmentations,
        "timestamp": str(datetime.datetime.now()),
    }
    save_annotations(annotations)

    # Save annotated image
    if boxes or segmentations:
        vis = draw_annotations(img["path"], boxes, labels, box_types, segmentations)
        vis_path = VISUAL_DIR / f"annotated_{img['filename']}"
        vis.save(vis_path)

    return jsonify({"success": True, "idx": idx, "total_annotated": len(annotations)})


@app.route("/api/annotated_image/<int:idx>")
def api_annotated_image(idx):
    """Return the annotated image (generated on the fly)."""
    if idx < 0 or idx >= len(IMAGE_LIST):
        return "Not found", 404
    annotations = load_annotations()
    str_idx = str(idx)
    ann = annotations.get(str_idx)
    if not ann or not ann.get("boxes"):
        path = IMAGE_LIST[idx]["path"]
        with open(path, "rb") as f:
            return Response(f.read(), mimetype="image/jpeg")
    vis = draw_annotations(IMAGE_LIST[idx]["path"],
                           ann["boxes"], ann["labels"])
    buf = io.BytesIO()
    vis.save(buf, format="JPEG", quality=92)
    buf.seek(0)
    return Response(buf.getvalue(), mimetype="image/jpeg")


@app.route("/api/clear/<int:idx>", methods=["DELETE"])
def api_clear(idx):
    """Clear annotations for an image."""
    if idx < 0 or idx >= len(IMAGE_LIST):
        return jsonify({"error": "Invalid index"}), 400
    annotations = load_annotations()
    str_idx = str(idx)
    if str_idx in annotations:
        del annotations[str_idx]
        save_annotations(annotations)
    # Delete cached annotated image
    vis_path = VISUAL_DIR / f"annotated_{IMAGE_LIST[idx]['filename']}"
    if vis_path.exists():
        vis_path.unlink()
    return jsonify({"success": True})


@app.route("/api/stats")
def api_stats():
    annotations = load_annotations()
    total = len(IMAGE_LIST)
    annotated = len(annotations)
    by_class = {}
    for ann in annotations.values():
        cls = ann.get("class", "Unknown")
        by_class[cls] = by_class.get(cls, 0) + 1
    return jsonify({
        "total": total,
        "annotated": annotated,
        "remaining": total - annotated,
        "progress_pct": round(annotated / total * 100, 1) if total else 0,
        "by_class": by_class,
    })


@app.route("/api/export")
def api_export():
    """Download all annotations as JSON."""
    annotations = load_annotations()
    return jsonify(annotations)


@app.route("/api/export_yolo")
def api_export_yolo():
    """Export all annotations in YOLO format as a zip file."""
    import tempfile, zipfile, shutil

    annotations = load_annotations()
    temp_dir = tempfile.mkdtemp()

    for str_idx, ann in annotations.items():
        boxes = ann.get("boxes", [])
        if not boxes:
            continue
        class_name = ann.get("class", "No Tumor")
        filename = ann.get("filename", f"image_{str_idx}")
        base_name = os.path.splitext(filename)[0]

        img_path = ann.get("path", "")
        try:
            with Image.open(img_path) as img:
                iw, ih = img.size
        except Exception:
            iw, ih = 512, 512

        txt_path = os.path.join(temp_dir, base_name + ".txt")
        with open(txt_path, "w") as f:
            for i, box in enumerate(boxes):
                label = ann.get("labels", [""])[i] if i < len(ann.get("labels", [])) else class_name
                cls = CLASSES.index(label) if label in CLASSES else 0
                box_type = ann.get("box_types", [])[i] if i < len(ann.get("box_types", [])) else "rectangle"

                if box_type == "polygon":
                    segs = ann.get("segmentations", [])
                    seg = segs[i] if i < len(segs) else {}
                    pts = seg.get("points", [])
                    if pts:
                        norm = " ".join(f"{px/iw:.6f} {py/ih:.6f}" for px, py in pts)
                        f.write(f"{cls} {norm}\n")
                        continue

                x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
                cx = ((x1 + x2) / 2) / iw
                cy = ((y1 + y2) / 2) / ih
                bw = (x2 - x1) / iw
                bh = (y2 - y1) / ih
                f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fn in os.listdir(temp_dir):
            if fn.endswith(".txt"):
                zf.write(os.path.join(temp_dir, fn), fn)
    shutil.rmtree(temp_dir, ignore_errors=True)
    zip_buf.seek(0)
    return Response(
        zip_buf.getvalue(),
        mimetype="application/zip",
        headers={
            "Content-Disposition": f'attachment;filename=yolo_annotations_{datetime.date.today()}.zip'
        },
    )


@app.route("/api/auto_annotate/<int:idx>", methods=["POST"])
def api_auto_annotate(idx):
    """Auto-annotate single image: YOLOv8 (primary) + U-Net seg."""
    if clf_model is None and yolo_model is None:
        return jsonify({"error": "No AI model loaded"}), 500
    if idx < 0 or idx >= len(IMAGE_LIST):
        return jsonify({"error": "Invalid index"}), 400
    try:
        result = _annotate_one(idx)
        return jsonify(result)
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


def _annotate_one(idx):
    """Core annotation logic for one image. Returns dict."""
    img_info = IMAGE_LIST[idx]
    img_path = img_info["path"]
    pil = Image.open(img_path).convert("RGB")
    orig_w, orig_h = pil.size

    new_boxes, new_labels, new_types, new_segs = [], [], [], []
    pred_class = img_info["class"]
    confidence = 0.0
    method = "none"

    # ── YOLOv8 (primary) ─────────────────────────────────────────────────────
    if yolo_model is not None:
        method = "yolov8"
        res = yolo_model.predict(pil, conf=0.20, verbose=False, imgsz=640)[0]
        if res.boxes is not None and len(res.boxes) > 0:
            for box in res.boxes:
                x1,y1,x2,y2 = [int(v) for v in box.xyxy[0].cpu().numpy()]
                conf   = round(float(box.conf[0]) * 100, 1)
                cls_id = int(box.cls[0])
                det_cls = CLASSES[cls_id] if cls_id < len(CLASSES) else img_info["class"]
                new_boxes.append({"x1":x1,"y1":y1,"x2":x2,"y2":y2})
                new_labels.append(det_cls); new_types.append("rectangle"); new_segs.append({})
                if conf > confidence:
                    pred_class = det_cls; confidence = conf

    # ── GradCAM fallback ──────────────────────────────────────────────────────
    if not new_boxes and clf_model is not None:
        method = "gradcam"
        tensor = clf_tf(pil).unsqueeze(0).to(DEVICE)
        cam, pred_idx, probs = gradcam.generate(tensor)
        pred_class = CLASSES[pred_idx]
        confidence = round(float(probs[pred_idx]) * 100, 1)
        cam_r = cv2.resize(cam, (orig_w, orig_h))
        binary = (cam_r >= 0.40).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            pts = np.concatenate(contours)
            x,y,w,h = cv2.boundingRect(pts)
            pad_x,pad_y = int(w*0.08), int(h*0.08)
            x1=max(0,x-pad_x); y1=max(0,y-pad_y)
            x2=min(orig_w,x+w+pad_x); y2=min(orig_h,y+h+pad_y)
        else:
            mx,my=orig_w//2,orig_h//2; pw,ph=int(orig_w*0.3),int(orig_h*0.3)
            x1,y1,x2,y2=mx-pw,my-ph,mx+pw,my+ph
        new_boxes.append({"x1":x1,"y1":y1,"x2":x2,"y2":y2})
        new_labels.append(pred_class); new_types.append("rectangle"); new_segs.append({})

    # ── U-Net segmentation ────────────────────────────────────────────────────
    has_seg = False
    if unet is not None and new_boxes:
        tensor_seg = seg_tf(pil).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            mask = unet(tensor_seg).squeeze().cpu().numpy()
        mask_r = cv2.resize(mask, (orig_w, orig_h))
        mask_bin = (mask_r > 0.5).astype(np.uint8)
        contours_seg, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_seg:
            largest = max(contours_seg, key=cv2.contourArea)
            epsilon = 0.02 * cv2.arcLength(largest, True)
            approx  = cv2.approxPolyDP(largest, epsilon, True)
            if len(approx) >= 3:
                seg_pts = [[int(p[0][0]),int(p[0][1])] for p in approx]
                xs=[p[0] for p in seg_pts]; ys=[p[1] for p in seg_pts]
                new_boxes.append({"x1":min(xs),"y1":min(ys),"x2":max(xs),"y2":max(ys)})
                new_labels.append(pred_class); new_types.append("polygon")
                new_segs.append({"points":seg_pts}); has_seg = True

    # ── Save ──────────────────────────────────────────────────────────────────
    annotations = load_annotations()
    annotations[str(idx)] = {
        "idx": idx, "filename": img_info["filename"],
        "class": img_info["class"], "split": img_info["split"],
        "path": img_info["path"], "boxes": new_boxes,
        "labels": new_labels, "box_types": new_types,
        "segmentations": new_segs, "auto_annotated": True,
        "method": method, "pred_class": pred_class,
        "confidence": confidence,
        "timestamp": str(datetime.datetime.now()),
    }
    save_annotations(annotations)

    vis = draw_annotations(img_path, new_boxes, new_labels, new_types, new_segs)
    vis.save(VISUAL_DIR / f"annotated_{img_info['filename']}")

    return {
        "success": True, "boxes": new_boxes, "labels": new_labels,
        "box_types": new_types, "segmentations": new_segs,
        "pred_class": pred_class, "confidence": confidence,
        "method": method, "has_seg": has_seg,
    }


@app.route("/api/auto_annotate_all", methods=["POST"])
def api_auto_annotate_all():
    """Auto-annotate ALL images. Body: {skip_done: true/false}"""
    if clf_model is None and yolo_model is None:
        return jsonify({"error": "No AI model loaded"}), 500

    data      = request.get_json() or {}
    skip_done = data.get("skip_done", True)
    existing  = load_annotations()

    total   = len(IMAGE_LIST)
    done    = 0
    skipped = 0
    failed  = 0

    for idx in range(total):
        if skip_done and str(idx) in existing:
            skipped += 1
            continue
        try:
            _annotate_one(idx)
            done += 1
        except Exception:
            failed += 1

    return jsonify({
        "success": True,
        "total":   total,
        "done":    done,
        "skipped": skipped,
        "failed":  failed,
        "message": f"✅ Done {done} | Skipped {skipped} | Failed {failed}",
    })


@app.route("/api/ai_status")
def api_ai_status():
    return jsonify({
        "classifier": clf_model is not None,
        "unet":       unet is not None,
        "yolo":       yolo_model is not None,
        "device":     str(DEVICE),
    })


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"  FundusAnnotator — AI-Powered Annotation Tool")
    print(f"  Dataset images : {len(IMAGE_LIST)}")
    print(f"  Classifier     : {'✅' if clf_model else '❌'}")
    print(f"  YOLOv8         : {'✅ mAP@0.5=98.55%' if yolo_model else '❌ not found'}")
    print(f"  U-Net          : {'✅' if unet else '❌ run Week4 first'}")
    print(f"  Open           : http://127.0.0.1:5050")
    app.run(host="127.0.0.1", port=5050, debug=True)

