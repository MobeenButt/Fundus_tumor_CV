"""
Fundus Tumor Classification - Prediction Web App (v94)
Upload any fundus image and get instant AI classification.
Run: python predict_app.py → http://127.0.0.1:5094
"""

import io
import json
import base64
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
from flask import Flask, render_template, request, jsonify

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.pth"
CLASSES_PATH = BASE_DIR / "classes.json"

# ── Load class names ──────────────────────────────────────────────────────────
with open(CLASSES_PATH) as f:
    CLASS_NAMES = json.load(f)

DISPLAY_NAMES = {
    "Choroidal Hemangioma (CH)":     "Choroidal Hemangioma",
    "Choroidal Osteoma (CO)":        "Choroidal Osteoma",
    "Normal":                         "Normal",
    "Retinal Capillary Hemangioma (RCH)": "Retinal Cap. Hemangioma",
    "Retinoblastoma (RB)":           "Retinoblastoma",
    "Uveal Melanoma (UM)":           "Uveal Melanoma",
}

CLASS_COLORS = {
    "Choroidal Hemangioma (CH)":     "#FF6B6B",
    "Choroidal Osteoma (CO)":        "#4ECDC4",
    "Normal":                         "#95E77D",
    "Retinal Capillary Hemangioma (RCH)": "#FFE66D",
    "Retinoblastoma (RB)":           "#FF85A2",
    "Uveal Melanoma (UM)":           "#5A67D8",
}

CLASS_INFO = {
    "Choroidal Hemangioma (CH)":     "Benign vascular tumor of the choroid layer.",
    "Choroidal Osteoma (CO)":        "Benign bone tumor of the choroid.",
    "Normal":                         "Normal fundus without any tumor.",
    "Retinal Capillary Hemangioma (RCH)": "Benign vascular lesion of the retina.",
    "Retinoblastoma (RB)":           "Malignant tumor of the retina (pediatric).",
    "Uveal Melanoma (UM)":           "Malignant melanoma of the uveal tract.",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224

# ── Load model ────────────────────────────────────────────────────────────────
def build_model(num_classes):
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model

model = build_model(len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE).eval()
print(f"✅ Model loaded on {DEVICE}")

# ── Transform ─────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── Predict function ──────────────────────────────────────────────────────────
def predict(pil_img):
    tensor = transform(pil_img.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze().cpu().numpy()
    results = [
        {
            "class":   cls,
            "display": DISPLAY_NAMES.get(cls, cls),
            "prob":    round(float(probs[i]) * 100, 2),
            "color":   CLASS_COLORS.get(cls, "#CCCCCC"),
            "info":    CLASS_INFO.get(cls, ""),
        }
        for i, cls in enumerate(CLASS_NAMES)
    ]
    results.sort(key=lambda x: x["prob"], reverse=True)
    return results

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates")

@app.route("/")
def index():
    return render_template("predict.html",
                           class_names=CLASS_NAMES,
                           class_colors=CLASS_COLORS,
                           display_names=DISPLAY_NAMES)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        return jsonify({"error": "Invalid file type. Use JPG or PNG."}), 400

    try:
        pil_img = Image.open(file.stream).convert("RGB")
    except Exception:
        return jsonify({"error": "Could not read image"}), 400

    # Run prediction
    results = predict(pil_img)

    # Encode image as base64 for display
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=90)
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    return jsonify({
        "success":    True,
        "prediction": results[0],   # top prediction
        "all":        results,       # all class probabilities
        "image_b64":  img_b64,
        "filename":   file.filename,
    })

if __name__ == "__main__":
    print("=" * 50)
    print("  Fundus Tumor - Prediction App (v94)")
    print(f"  Model : EfficientNet-B0")
    print(f"  Device: {DEVICE}")
    print(f"  Open  : http://127.0.0.1:5094")
    print("=" * 50)
    app.run(host="127.0.0.1", port=5094, debug=False)
