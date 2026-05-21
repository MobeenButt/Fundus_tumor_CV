import os
import json
import io
import base64
import datetime
from pathlib import Path

from flask import (
    Flask, render_template, send_file, request, jsonify, url_for, Response,
)
from PIL import Image, ImageDraw

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
ANNOTATIONS_FILE = BASE_DIR / "annotations_flask.json"
VISUAL_DIR = BASE_DIR / "annotated_images"
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

app = Flask(__name__)
VISUAL_DIR.mkdir(exist_ok=True)




# ── Image scanner ─────────────────────────────────────────────────────────────
def scan_images(data_dir=None):
    if data_dir is None:
        data_dir = BASE_DIR.parent  # dataset is one level above flask_annotator/
    images = []
    for split in ("Training", "Testing"):
        for folder_name, class_name in FOLDER_TO_CLASS.items():
            class_path = Path(data_dir) / split / folder_name
            if class_path.exists():
                for f in sorted(class_path.iterdir()):
                    if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                        images.append({
                            "path": str(f.resolve()),
                            "class": class_name,
                            "split": split,
                            "filename": f.name,
                        })
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


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"  FundusAnnotator - Fundus Tumor Annotation Tool")
    print(f"  Dataset images: {len(IMAGE_LIST)}")
    print(f"  Annotations file: {ANNOTATIONS_FILE}")
    print(f"  Open: http://127.0.0.1:5050")
    app.run(host="127.0.0.1", port=5050, debug=True)
