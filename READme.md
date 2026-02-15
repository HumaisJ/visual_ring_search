### 1. Project README File

This is ready to be copied and pasted into a new file named `README.md` in the root of your `visual_ring_search` project folder.

````markdown
# Visual Ring Search Engine

This project is a high-accuracy visual search engine designed to find jewelry rings from a catalog based on the similarity of their design, intelligently ignoring the gemstones. It uses a multi-stage AI pipeline, including a custom-trained object detector, to achieve robust and accurate results from any input image.

## Core Features

-   **AI-Powered Ring Detection:** Utilizes a custom-trained **YOLOv8** model to automatically detect and crop rings from raw input images, making the system robust to noisy backgrounds and varied compositions.
-   **Advanced Image Processing:** Once a ring is detected, it undergoes a sophisticated preprocessing pipeline:
    1.  **Background Removal:** Isolates the ring from its background using the `rembg` library.
    2.  **Stone Segmentation:** Masks out gemstones based on color and saturation, focusing the analysis purely on the ring's metalwork and design.
    3.  **Design Normalization:** Extracts the high-frequency design details of the ring and normalizes them for lighting and contrast variations.
-   **Deep Feature Extraction:** Employs a pre-trained **EfficientNetB0** model to convert the normalized ring design into a feature vector (a numerical "fingerprint").
-   **High-Speed Similarity Search:** Indexes all catalog items using their feature vectors and finds the most similar designs to a query image using a `NearestNeighbors` search, providing near-instantaneous results.
-   **Interactive Web Interface:** A user-friendly web application built with **Streamlit** allows for easy catalog management and intuitive visual searching.

## Project Structure

```
visual_ring_search/
├── dataset/
│   ├── 01_raw/               # Raw, unprocessed images for training the detector
│   ├── 02_labeled_rings/     # Labeled data (from Roboflow) for the YOLO model
│   └── 03_design_catalog/    # Clean images of catalog items
├── models/
│   ├── catalog_index_robust.pkl # The saved, indexed catalog of feature vectors
│   └── ring_detector_run/    # The output from the YOLOv8 training process
├── notebooks/
│   ├── train_detector.py     # Script to train the YOLOv8 Ring Detective
│   ├── detector_app.py       # Standalone Streamlit app for testing the detector
│   └── ...
├── app/
│   ├── engine.py             # Core backend logic for the entire ML pipeline
│   ├── app.py                # Main Streamlit application for the search engine
│   └── requirements.txt      # Python dependencies
└── README.md                 # This file
```

## How to Use

1.  **Setup the Environment:**
    -   Ensure you have Python 3.9+ installed.
    -   Navigate to the `app/` directory and install all required libraries:
        ```bash
        pip install -r requirements.txt
        ```

2.  **Train the Ring Detector (if not already done):**
    -   Label your images in the `dataset/01_raw` folder using a tool like [Roboflow](https://roboflow.com) and export them in `YOLOv8` format to `dataset/02_labeled_rings`.
    -   Navigate to the `notebooks/` directory and run the training script:
        ```bash
        python train_detector.py
        ```
    -   This will generate a `best.pt` model file. Ensure the `DETECTOR_MODEL_PATH` in `app/engine.py` points to this file's absolute path.

3.  **Launch the Main Search Engine:**
    -   Navigate to the `app/` directory.
    -   Run the Streamlit application:
        ```bash
        streamlit run app.py
        ```
    -   The first time you launch, click the **"Build / Rebuild Catalog Index"** button in the sidebar. This will process all images in the `dataset/03_design_catalog` and save the persistent index.
    -   Once the catalog is built, you can upload any image to search for similar rings.
````
