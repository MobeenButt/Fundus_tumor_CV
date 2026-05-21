# Fundus Tumor Classification - Model v94

This folder contains a trained **EfficientNet-B0** model for fundus tumor classification.

## 📦 Contents
- `best_model.pth` - Trained model weights
- `classes.json` - Class names mapping
- `test_app.py` - Interactive GUI test application
- `predict_app.py` - Flask web app for image upload & prediction
- `evaluate.py` - Evaluation script (generates metrics & confusion matrix)
- `requirements.txt` - Python dependencies
- `templates/` - HTML templates for Flask app
- `metrics.txt` - Training metrics
- `history.json` - Training history
- `confusion_matrix.png` - Model confusion matrix
- `training_curves.png` - Training/validation curves

## 🎯 Classes
The model classifies fundus images into 6 categories:
1. **Choroidal Hemangioma (CH)** - Benign vascular tumor of the choroid
2. **Choroidal Osteoma (CO)** - Benign bone tumor of the choroid
3. **Normal** - Normal fundus without tumor
4. **Retinal Capillary Hemangioma (RCH)** - Benign vascular lesion of retina
5. **Retinoblastoma (RB)** - Malignant tumor of the retina
6. **Uveal Melanoma (UM)** - Malignant melanoma of uveal tract

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Test App
```bash
python test_app.py
```

A graphical window will open with options to:
- **📷 Test Single Image** - Click to select an image file
- **📂 Test Folder of Images** - Click to select a folder containing images
- **❌ Exit** - Close the application

## 📝 Usage

### Testing a Single Image
1. Click **"📷 Test Single Image"** button
2. A file dialog opens - browse and select an image (JPG, PNG, BMP)
3. Results display instantly with:
   - Top prediction and confidence score
   - All class probabilities
   - Medical information about the condition

### Testing Multiple Images
1. Click **"📂 Test Folder of Images"** button
2. A folder dialog opens - select your test folder
3. The app processes all images and displays:
   - Individual predictions with confidence scores
   - Per-class accuracy (if folder structure matches class names)
   - Overall accuracy summary
   - Detailed JSON report saved automatically

## � Evaluation Script

For comprehensive model evaluation on the full test dataset with detailed metrics:

```bash
python evaluate.py
```

This script will:
- ✅ Load the test dataset from `finals/data/Testing/`
- 🎯 Run inference on all test images
- 📊 Calculate accuracy and per-class metrics
- 📈 Generate confusion matrix visualization
- 📄 Save detailed metrics report to `results/metrics.txt`

### Output Files:
- `results/confusion_matrix.png` - Heatmap of predictions vs ground truth
- `results/metrics.txt` - Detailed accuracy, precision, recall, F1-score
## 🌐 Web Application (Flask)

For a user-friendly web interface with drag-and-drop image upload:

```bash
python predict_app.py
```

Then open your browser to: **http://127.0.0.1:5094**

### Features:
- 📷 Drag & drop or click to upload images
- 🎯 Instant predictions with confidence scores
- 📊 Visual probability bars for all 6 classes
- 📝 Medical information for predictions
- 🖼️ Image preview in browser
- 📱 Responsive mobile-friendly design

### Usage:
1. Open http://127.0.0.1:5094 in your browser
2. Upload a fundus image (JPG, PNG, BMP)
3. See instant predictions with confidence scores
4. View detailed results and medical information
## �📁 Expected Folder Structure (for accuracy calculation)
```
test_images/
├── Choroidal Hemangioma (CH)/
│   ├── image1.jpg
│   └── image2.jpg
├── Choroidal Osteoma (CO)/
│   ├── image3.jpg
│   └── ...
├── Normal/
│   └── ...
└── (other class folders)
```

## 📊 Output Display

### Single Image Prediction (in GUI):
```
======================================================================
🎯 Top Prediction: Uveal Melanoma (UM)
   Confidence: 95.47%

All Predictions:
Class                                    Confidence
---------------------------------------- ---------------
Uveal Melanoma (UM)                          95.47%
Choroidal Hemangioma (CH)                     3.21%
Retinoblastoma (RB)                           1.15%
...

📝 Info: Malignant melanoma of the uveal tract.
======================================================================
```

A success popup also appears showing the top prediction and confidence.

### Batch Folder Testing (in GUI):
```
📊 Summary by Class:
======================================================================
Choroidal Hemangioma (CH)                  8/10 ( 80.00%)
Choroidal Osteoma (CO)                     9/10 ( 90.00%)
Normal                                    10/10 (100.00%)
Retinal Capillary Hemangioma (RCH)         7/10 ( 70.00%)
Retinoblastoma (RB)                        8/10 ( 80.00%)
Uveal Melanoma (UM)                        9/10 ( 90.00%)
----------------------------------------------------------------------
Overall Accuracy                           51/60 ( 85.00%)
======================================================================
```

A success popup shows the overall accuracy and confirms results were saved.

## 🖥️ Device Detection
- **GPU**: If CUDA is available, uses GPU for faster inference
- **CPU**: Falls back to CPU if GPU is not available

## 📌 Notes
- Image size is automatically resized to 224×224 pixels
- Images are normalized using ImageNet statistics
- Confidence scores represent probability for each class
- Detailed JSON reports are saved for batch testing

## 🔧 Troubleshooting

**Error: "Model file not found"**
- Make sure `best_model.pth` is in the same folder as `test_app.py`, `predict_app.py`, or `evaluate.py`

**Error: "Classes file not found"**
- Make sure `classes.json` is in the same folder as the script

**Error: "Test directory not found" (for evaluate.py)**
- Make sure the test dataset exists at `finals/data/Testing/`
- Each class should have a subfolder with test images

**Flask app won't start (predict_app.py)**
- Make sure Flask is installed: `pip install -r requirements.txt`
- Check if port 5094 is in use: Try `python predict_app.py` again
- Windows firewall may block it: Allow Python through firewall

**Slow inference**
- Use GPU for faster processing
- On Colab: Runtime → Change runtime type → T4 GPU

**Out of memory error**
- Reduce batch size or test images one at a time with test_app.py or predict_app.py
- For evaluate.py, reduce batch_size parameter in DataLoader

**"templates not found" error**
- Make sure `templates/` folder exists in the same directory as `predict_app.py`
- Ensure `templates/predict.html` file is present

---
**Model Version**: 94  
**Architecture**: EfficientNet-B0  
**Date Created**: 2024

## 📋 Quick Summary

| Task | Script | Command |
|------|--------|---------|
| **Quick GUI Testing** | `test_app.py` | `python test_app.py` |
| **Web Interface** | `predict_app.py` | `python predict_app.py` → http://127.0.0.1:5094 |
| **Batch Evaluation** | `evaluate.py` | `python evaluate.py` |
| **Single Prediction** | Python Code | See code examples in `predict_app.py` |
