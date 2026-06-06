# Implementation Tools Used & Working

## Tools & Libraries

| Tool/Library | Version | Purpose |
|---|---|---|
| **PyTorch** | 2.1.0 | Deep learning framework for U-Net model definition, training, and inference |
| **TorchVision** | 0.16.0 | Image transforms and utilities |
| **OpenCV** | 4.8.1.78 | Image I/O, morphological operations (closing/opening), contour detection, resizing |
| **Pillow** | 10.0.1 | Image loading and basic manipulation |
| **NumPy** | 1.24.3 | Array operations and numerical computations |
| **Matplotlib** | 3.8.1 | All visualizations (training curves, metrics bar charts, segmentation overlays) |
| **scikit-learn** | 1.3.2 | Metrics computation (IoU, Dice, Pixel Accuracy) |
| **tqdm** | 4.66.1 | Progress bars during training and evaluation |
| **Google Colab** | — | GPU-accelerated training runtime (Tesla T4) |

## How It Works

### 1. Pseudo-Mask Generation (GradCAM)
- The pre-trained EfficientNet-B0 classifier (`model/best_model.pth`) is used as a backbone.
- GradCAM hooks onto the last convolutional layer (`model.features[-1]`) to generate activation heatmaps.
- Heatmaps are thresholded at **0.35** to produce binary pseudo-masks.
- Morphological closing and opening (7x7 ellipse kernel) clean noise from the masks.
- Masks are generated for all training and testing images to serve as ground truth for U-Net.

### 2. U-Net Architecture
- Custom U-Net built from scratch with PyTorch.
- **Encoder:** DoubleConv blocks with channel depths [64, 128, 256, 512]
- **Bottleneck:** 1024 channels
- **Decoder:** Transposed convolutions mirrored with skip connections
- **Output:** 1x1 convolution + Sigmoid activation
- **Total parameters:** ~31M
- Each DoubleConv: Conv2D -> BatchNorm -> ReLU -> Conv2D -> BatchNorm -> ReLU

### 3. Training Configuration
- **Input size:** 256x256 pixels
- **Batch size:** 8
- **Loss function:** Dice + BCE (combined)
- **Optimizer:** AdamW
- **Learning rate scheduler:** CosineAnnealing
- **Early stopping:** Patience of 7 epochs
- **Augmentations:** Random horizontal and vertical flips
- **Epochs trained:** 6 (early stopped)

### 4. Evaluation Pipeline
- IoU, Dice Score, and Pixel Accuracy computed per sample.
- Mean metrics aggregated across the entire test set.
- Visualizations: training curves, metrics bar chart, sample predictions with overlays.

## Working

The pipeline runs end-to-end in Google Colab with GPU acceleration. The single notebook `Week4_Segmentation_Colab.ipynb` covers:

1. Dataset download from Kaggle
2. GradCAM pseudo-mask generation
3. U-Net dataset creation and loading
4. Model training with early stopping
5. Evaluation and metric computation
6. Visualization generation
7. ZIP export of all results

All outputs (model weights, metrics JSON, visualizations) are saved to this directory.
