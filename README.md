# Fundus Tumor Detection — Week 3 (YOLOv8)

## Approach

Since no ground-truth bounding boxes were available for the fundus tumor dataset, we used a **GradCAM-based auto-annotation pipeline** to generate pseudo-labels for object detection.

1. **Classification backbone**: A trained EfficientNet-B0 (`best_model.pth`) classifies fundus images into 6 classes (CH, CO, Normal, RCH, RB, UM) with ~94% accuracy.
2. **GradCAM heatmaps**: GradCAM extracts class-discriminative activation maps from the final convolutional layer of EfficientNet-B0, highlighting tumor regions.
3. **Bounding box generation**: Heatmaps are thresholded and contour-extracted to produce YOLO-format labels (normalized `class_id cx cy w h`). This produced **2,031 annotations** (1,623 train + 408 val).
4. **YOLOv8n training**: The nano variant of YOLOv8 was trained for 50 epochs at 640×640 resolution using the auto-generated labels.

## Results

| Metric     | Value  |
|------------|--------|
| mAP@0.5    | 98.55% |
| mAP@0.5:0.95 | 79.0% |
| Precision  | 96.91% |
| Recall     | 96.39% |

## Tools Used

- **PyTorch / torchvision** — classification model and GradCAM
- **Ultralytics YOLOv8** — object detection training & inference
- **OpenCV** — contour extraction for bounding box generation
- **Google Colab** — GPU-accelerated training environment
- **Flask** — web deployment combining YOLO detection with classification

## Key Files

| File | Purpose |
|------|---------|
| `generate_annotations.py` | Auto-generates YOLO labels via GradCAM |
| `annotations/dataset.yaml` | YOLOv8 dataset configuration |
| `annotations/labels/train/` | 1,623 YOLO training labels |
| `annotations/labels/val/` | 408 YOLO validation labels |
| `Week3_YOLO_Results/best.pt` | Trained YOLOv8n weights (98.55% mAP) |
| `Week3_YOLO_Results/yolo_metrics.json` | Per-class evaluation metrics |
| `Week3_YOLO_Colab.ipynb` | Full training pipeline notebook |
