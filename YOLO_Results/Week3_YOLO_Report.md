# Week 3 — YOLOv8 Object Detection Report

## Objective

Train a YOLOv8 object detector on ultra-wide fundus eye images to detect and classify six types of ocular conditions/tumors using automatically generated bounding box labels from a pretrained classifier + GradCAM.

**Classes (6):** Choroidal Hemangioma (CH), Choroidal Osteoma (CO), Normal, Retinal Capillary Hemangioma (RCH), Retinoblastoma (RB), Uveal Melanoma (UM)

---

## Approach

### 1. Dataset Acquisition

- **Source:** Kaggle — [nikitamanaenkov/ultra-wide-fundus-images-for-tumor-diagnosis](https://kaggle.com/datasets/nikitamanaenkov/ultra-wide-fundus-images-for-tumor-diagnosis)
- Kaggle API is configured with a token, and the dataset is downloaded and extracted into `/content/dataset/`.

### 2. Classifier & GradCAM Label Generation

Since the dataset provides only image-level class labels (no bounding boxes), we generate pseudo-labels using an **EfficientNet-B0 classifier** that was previously trained on the same fundus images:

1. **Classifier:** EfficientNet-B0 with a dropout + linear head for 6 classes.
2. **GradCAM:** Hooks into the last feature layer; heatmaps highlight regions the classifier finds most discriminative.
3. **Heatmap → Bounding Box:** CAM is resized to original image dimensions, thresholded (`>=0.40`), and contours are extracted. The bounding rect (with 10% padding) becomes the YOLO-format label.

### 3. YOLOv8 Training

- **Model:** `yolov8n.pt` (nano variant) — fast to train on Colab free tier.
- **Image Size:** 640×640
- **Epochs:** 50 (with early stopping patience=10)
- **Batch Size:** 16
- The auto-labeled dataset is organized into `images/train`, `images/val`, `labels/train`, `labels/val` with a YAML data config for Ultralytics.

### 4. Evaluation

The trained model is evaluated on the test set using standard COCO metrics:
- mAP@0.5
- mAP@0.5:0.95
- Precision & Recall
- Per-class AP@0.5

### 5. Visualization & Export

- Sample predictions are visualized per class (2 per class) with bounding boxes + confidence scores.
- Training plots, confusion matrix, PR/F1 curves are saved.
- A `yolo_metrics.json` file and a ZIP archive of all results are generated for download.

---

## Tools & Libraries

| Tool | Purpose |
|---|---|
| **Kaggle API** | Dataset download |
| **PyTorch** | EfficientNet-B0 classifier + GradCAM hooks |
| **TorchVision** | Image transforms (Resize, Normalize) |
| **OpenCV** | CAM resizing, contour detection, bounding rect |
| **Ultralytics YOLOv8** | Object detection model (train, val, predict) |
| **NumPy** | Array operations |
| **PIL / Pillow** | Image I/O |
| **Matplotlib** | Visualizing predictions grid |
| **tqdm** | Progress bars during labeling |
| **Google Colab** | Runtime (GPU) |

---

## Pipeline Summary

```
Kaggle Dataset (class labels only)
        ↓
EfficientNet-B0 + GradCAM → CAM heatmaps
        ↓
Heatmap thresholding + contour detection → YOLO bounding boxes
        ↓
YOLOv8n training (50 epochs, 640×640)
        ↓
Evaluation (mAP@0.5, mAP@0.5:0.95, Precision, Recall)
        ↓
Visualization + metrics export + ZIP download
```

---

## Results

| Metric | Value |
|---|---|
| mAP@0.5 | *(computed at runtime)* |
| mAP@0.5:0.95 | *(computed at runtime)* |
| Precision | *(computed at runtime)* |
| Recall | *(computed at runtime)* |

*Actual metric values are populated after running the notebook.*
