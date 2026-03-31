# Visual Ring Search Engine

This project is a high-accuracy visual search engine designed to find jewelry rings from a catalog based on the similarity of their design, intelligently ignoring gemstones. It implements a **multi-stage deep learning pipeline** combining object detection, image processing, feature extraction, and similarity search.

---

## 🧠 ML Pipeline Overview

The system is built as a **hybrid computer vision pipeline**, combining:

* **Custom Object Detection (YOLOv8)**
* **Image Preprocessing & Segmentation**
* **Deep Feature Extraction (EfficientNetB0)**
* **Similarity Search (k-NN / Nearest Neighbors)**

### Flow:

```
Input Image → Ring Detection → Preprocessing → Feature Extraction → Similarity Matching → Results
```

---

## 🔍 Core Features

### 1. AI-Powered Ring Detection

* Uses a custom-trained **YOLOv8 (Ultralytics)** model.
* Type: **Supervised Deep Learning (Object Detection)**
* Base model: `yolov8n.pt` (pretrained on COCO dataset)
* Fine-tuned using **transfer learning** on a custom ring dataset labeled via Roboflow.

**What it does:**

* Detects rings in an image
* Outputs:

  * Bounding boxes
  * Confidence scores
  * Class label ("ring")

---

### 2. Training Details (YOLOv8)

* **Architecture:** CNN-based detector (Backbone + Neck + Detection Head)
* **Learning Type:** Supervised learning
* **Optimization:** Gradient descent (Adam/SGD internally)
* **Epochs:** 75
* **Image Size:** 640×640

#### Loss Functions Used

YOLOv8 optimizes a **multi-part loss function**:

* **Box Loss:** Measures bounding box accuracy (IoU-based)
* **Classification Loss:** Ensures correct object classification
* **Objectness Loss:** Detects whether an object exists in a region

```
Total Loss = Box Loss + Classification Loss + Objectness Loss
```

The entire neural network acts as the **hypothesis function**, learning to map images → object predictions.

---

### 3. Dataset & Labeling

* Images were **manually labeled using Roboflow**
* Format: YOLOv8 annotation format
* Structure:

  * Bounding boxes around rings
  * Single class: `ring`

This ensures the model learns:

* Shape
* Edges
* Structural features of rings (not background noise)

---

### 4. Advanced Image Processing Pipeline

After detection, the system refines the ring image:

#### a. Background Removal

* Library: `rembg`
* Removes irrelevant background pixels

#### b. Stone Segmentation

* Masks gemstones using:

  * Color filtering
  * Saturation thresholds
* Focus: metal band design only

#### c. Design Normalization

* Extracts high-frequency features
* Normalizes:

  * Lighting
  * Contrast
  * Texture

---

### 5. Deep Feature Extraction

* Model: **EfficientNetB0 (pretrained CNN)**
* Type: **Transfer Learning (Feature Extraction)**

**Purpose:**
Convert processed ring images into numerical feature vectors:

```
Image → Feature Vector (embedding)
```

This vector acts as a **design fingerprint**.

---

### 6. Similarity Search Engine

* Algorithm: **Nearest Neighbors (k-NN)**
* Library: `sklearn.neighbors`

**How it works:**

* All catalog images are converted into feature vectors
* Stored in an index
* Query image vector is compared using distance metrics

Typical metric:

* Euclidean distance or cosine similarity

```
Closest vectors → Most visually similar rings
```

---

### 7. Interactive Web Interface

* Built with **Streamlit**
* Features:

  * Upload query image
  * View detected rings
  * See top matching designs
  * Build/rebuild catalog index

---

## 📂 Project Structure

```
visual_ring_search/
├── dataset/
│   ├── 01_raw/               # Raw images for training
│   ├── 02_labeled_rings/     # Roboflow-labeled YOLO dataset
│   └── 03_design_catalog/    # Clean catalog images
├── models/
│   ├── catalog_index_robust.pkl # Stored feature vectors
│   └── ring_detector_run/    # YOLO training output
├── notebooks/
│   ├── train_detector.py     # YOLOv8 training script
│   ├── detector_app.py       # Detector testing app
│   └── ...
├── app/
│   ├── engine.py             # Core ML pipeline logic
│   ├── app.py                # Main Streamlit UI
│   └── requirements.txt
└── README.md
└── runtime.txt
```

---

## ⚙️ How to Use

### 1. Setup Environment

```bash
pip install -r requirements.txt
```

---

### 2. Train the Ring Detector

* Label images using Roboflow
* Export in YOLOv8 format

```bash
python train_detector.py
```

Output:

```
models/ring_detector_run/weights/best.pt
```

---

### 3. Launch the Application

```bash
streamlit run app.py
```

* Build catalog index (first run)
* Upload image
* Get visually similar rings

---

## 🧩 Key Concepts Demonstrated

* Object Detection (YOLOv8)
* Transfer Learning
* Feature Embedding
* Image Segmentation
* Similarity Search (k-NN)
* End-to-End ML Pipeline Design

---

## 🧠 What Makes This Project Unique

* Focuses on **design similarity**, not just object detection
* Explicitly removes gemstones to avoid misleading features
* Combines **multiple ML models into one pipeline**
* Uses both **deep learning + classical ML (k-NN)**

---

## ⚠️ Limitations

* Performance depends on dataset quality
* YOLOv8n trades accuracy for speed (can be improved with larger models)
* Stone segmentation is heuristic-based (not learned)

---
