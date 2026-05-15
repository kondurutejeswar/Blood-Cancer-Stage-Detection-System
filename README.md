# 🩸 AI-Based Blood Cancer Stage Detection Using Deep Learning

> An end-to-end deep learning system that classifies microscopic blood cell images into four leukemia stages with high accuracy — deployed as a real-time Flask web application.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-3.x-black?style=flat-square&logo=flask)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Demo](#-demo)
- [Classes](#-classification-categories)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [Model Architecture](#-model-architecture)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [Tech Stack](#-tech-stack)
- [Author](#-author)

---

## 🧠 Overview

Leukemia (blood cancer) affects over **400,000 people annually** worldwide. Early-stage detection can push survival rates above **85%**, while advanced-stage diagnosis drops it below **30%**.

This project automates blood cancer stage classification using a custom **Convolutional Neural Network (CNN)** trained on microscopic blood smear images. The trained model is served through a **Flask web application** where users can upload an image and receive an instant prediction with confidence scores for all four stages.

---

## 🎬 Demo

```
Upload Blood Smear Image → CNN Inference → Stage Prediction + Confidence %
```

| Page | Description |
|------|-------------|
| `/` | Home — project overview and class descriptions |
| `/predict` | Upload a blood smear image |
| `/upload` | POST endpoint — runs CNN inference |
| `/about` | About the project |

---

## 🔬 Classification Categories

| Stage | Folder | Clinical Meaning | Risk |
|-------|--------|-----------------|------|
| **Benign** | `benign/` | Normal healthy blood cells | 🟢 Low |
| **Early Stage** | `early/` | Initial abnormal morphology — highly treatable | 🟡 Moderate |
| **Pre-Cancerous** | `pre/` | Dysplastic cells with high transformation risk | 🟠 High |
| **Pro-Cancerous** | `pro/` | Fully malignant — urgent intervention needed | 🔴 Critical |

---

## 📁 Project Structure

```
blood_cancer_detection/
│
├── dataset/
│   ├── benign/                  ← Normal blood cell images
│   ├── early/                   ← Early stage leukemia images
│   ├── pre/                     ← Pre-cancerous stage images
│   └── pro/                     ← Pro-cancerous stage images
│
├── model_dump/
│   └── model.keras              ← Saved trained CNN model
│
├── notebooks/
│   └── model_training.ipynb     ← Full training notebook
│
├── static/
│   ├── uploads/                 ← User-uploaded images (runtime)
│   ├── style.css                ← Global stylesheet
│   └── accuracy_graph.png       ← Generated after training
│
├── templates/
│   ├── index.html               ← Home page
│   ├── predict.html             ← Upload page
│   ├── results.html             ← Prediction results page
│   └── about.html               ← About page
│
├── app.py                       ← Flask web application
├── main.py                      ← CNN training script
├── requirements.txt             ← Python dependencies
└── README.md
```

---

## ⚙️ How It Works

### Training Pipeline (`main.py`)

```
Raw Images (dataset/)
        │
        ▼
ImageDataGenerator
  ├── rescale = 1./255          (normalize pixels to [0, 1])
  ├── validation_split = 0.2   (80% train / 20% validation)
  └── target_size = (128, 128) (resize all images)
        │
        ▼
CNN Model Training (10 epochs, Adam, Categorical Crossentropy)
        │
        ▼
model.save("model_dump/model.keras")
        │
        ▼
accuracy_graph.png saved to static/
```

### Inference Pipeline (`app.py`)

```
User uploads image via browser
        │
        ▼
POST /upload → Flask receives file
        │
        ▼
secure_filename()        (sanitize filename)
image.load_img()         (load + resize to 128×128)
img_to_array() / 255.0   (normalize)
np.expand_dims(axis=0)   (add batch dimension)
        │
        ▼
model.predict()
        │
        ▼
np.argmax() → predicted class
Softmax probabilities → confidence scores for all 4 classes
        │
        ▼
render_template('results.html', predicted_class, confidence, all_predictions)
```

---

## 🏗️ Model Architecture

```
Input (128 × 128 × 3)
    │
    ├── Conv2D(32, 3×3, ReLU)  →  MaxPooling2D(2×2)   [Low-level features]
    ├── Conv2D(64, 3×3, ReLU)  →  MaxPooling2D(2×2)   [Mid-level features]
    ├── Conv2D(128, 3×3, ReLU) →  MaxPooling2D(2×2)   [High-level features]
    │
    ├── Flatten  →  Dense(128, ReLU)
    │
    └── Dense(4, Softmax)
         │
         └── [Benign, Early Stage, Pre-Cancerous, Pro-Cancerous]
```

| Parameter | Value |
|-----------|-------|
| Total Parameters | ~3.4 Million |
| Optimizer | Adam (lr = 0.001) |
| Loss Function | Categorical Crossentropy |
| Epochs | 10 |
| Batch Size | 32 |
| Input Size | 128 × 128 × 3 |
| Output | 4-class Softmax probabilities |

---

## 📊 Dataset

The dataset follows the standard `ImageDataGenerator` directory format:

```
dataset/
  benign/   → Normal blood cell images
  early/    → Early stage leukemia
  pre/      → Pre-cancerous
  pro/      → Pro-cancerous
```

**Preprocessing applied:**
- Pixel normalization: `[0, 255]` → `[0, 1]`
- Image resize: `128 × 128` pixels
- Train / Validation split: `80% / 20%`
- Color mode: RGB

> ⚠️ The dataset is not included in this repository. Place your images in the `dataset/` folder following the structure above before training.

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/blood-cancer-detection.git
cd blood-cancer-detection
```

### 2. Create a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🖥️ Usage

### Step 1 — Train the model

> Skip this step if you already have `model_dump/model.keras`

```bash
python main.py
```

This will:
- Load and preprocess images from `dataset/`
- Train the CNN for 10 epochs
- Save the model to `model_dump/model.keras`
- Save the accuracy graph to `static/accuracy_graph.png`

### Step 2 — Run the web application

```bash
python app.py
```

### Step 3 — Open in browser

```
http://127.0.0.1:5000
```

Upload a blood smear image (`.jpg`, `.jpeg`, or `.png`) and get an instant prediction.

---

## 📈 Results

| Metric | Value |
|--------|-------|
| Training Accuracy | _(insert from your output)_ |
| Validation Accuracy | _(insert from your output)_ |
| Training Loss | _(insert from your output)_ |
| Validation Loss | _(insert from your output)_ |
| Inference Speed | < 500ms per image |

> Replace the placeholder values above with your actual training results from `main.py` output.

The accuracy graph is automatically saved to `static/accuracy_graph.png` after training and displayed on the results page.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Deep Learning | TensorFlow 2.x / Keras |
| Web Framework | Flask |
| Image Processing | Pillow, OpenCV |
| Data Processing | NumPy |
| Visualization | Matplotlib |
| Security | Werkzeug (`secure_filename`) |
| Frontend | HTML5, CSS3, Jinja2 |

---

## 📦 Requirements

```txt
tensorflow
flask
werkzeug
numpy
matplotlib
Pillow
opencv-python
scikit-learn
```

Install all at once:

```bash
pip install -r requirements.txt
```

---

## 🔒 Security Notes

- All uploaded filenames are sanitized using `werkzeug.utils.secure_filename()` to prevent directory traversal attacks.
- Only `.png`, `.jpg`, and `.jpeg` file extensions are accepted.
- Set `debug=False` in `app.py` before deploying to production.

---

## 🔭 Future Enhancements

- [ ] Transfer Learning — EfficientNetB4 / ResNet50 for higher accuracy
- [ ] Grad-CAM — Visual heatmaps showing which regions the CNN focused on
- [ ] Data Augmentation — Rotation, flip, zoom for better generalization
- [ ] Batch Normalization + Dropout — Reduce overfitting
- [ ] Docker containerization for cloud deployment
- [ ] ONNX export for mobile inference

---

## 👤 Author

**K. Tejeswar**
Roll No: `232H5A0507`
B.Tech — Computer Science & Engineering
Audisankara Institute of Technology (A), Gudur

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

> ⚠️ **Disclaimer:** This system is developed for academic and research purposes only. It is **not** a substitute for professional medical diagnosis. Always consult a qualified healthcare professional for clinical decisions.
