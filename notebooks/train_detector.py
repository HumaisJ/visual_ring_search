from ultralytics import YOLO
import os

# --- CONFIGURATION ---
# IMPORTANT: You MUST change this path to match where your data.yaml file is located!
# To get the full path:
# 1. Go to your `dataset/02_labeled_rings/` folder.
# 2. Right-click on the `data.yaml` file and choose "Copy Path" or "Copy Absolute Path".
# 3. Paste it below inside the quotes.
# Example for Windows: 'C:\\Users\\YourName\\Desktop\\visual_ring_search\\dataset\\02_labeled_rings\\data.yaml'
# Example for Mac/Linux: '/Users/YourName/Desktop/visual_ring_search/dataset/02_labeled_rings/data.yaml'
DATA_YAML_PATH = 'E:\\Punjab University\\MoblieAppDev\\visual_ring_search\\dataset\\02_labeled_rings\\data.yaml'

# Training parameters
EPOCHS = 75  # How many times to train on the full dataset. 50-100 is a good start.
IMAGE_SIZE = 640 # The size of images to train on. 640 is standard for YOLO.


def main():
    """
    This script trains a YOLOv8 object detection model.
    """
    print("--- Starting YOLOv8 Ring Detection Training ---")

    # Check if the data.yaml path is valid
    if not os.path.exists(DATA_YAML_PATH):
        print(f"ERROR: The path to data.yaml is incorrect! Please fix it.")
        print(f"Current path: {DATA_YAML_PATH}")
        return

    # Load a pre-trained YOLOv8 model (yolov8n.pt is small and fast)
    # The model will download automatically on first use.
    model = YOLO('yolov8n.pt')

    print(f"Training with the following parameters:")
    print(f"  - Data config: {DATA_YAML_PATH}")
    print(f"  - Epochs: {EPOCHS}")
    print(f"  - Image Size: {IMAGE_SIZE}")
    print("-" * 20)

    # Start the training!
    # The results will be saved in a new `runs` folder in your current directory.
    model.train(
        data=DATA_YAML_PATH,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        project='../models/', # Save results in the main /models folder
        name='ring_detector_run' # Name of the subfolder for this run
    )

    print("-" * 20)
    print("--- Training Complete! ---")
    print("Your trained model is saved in the `visual_ring_search/models/ring_detector_run/weights/` folder.")
    print("The best performing model is named `best.pt`. This is your Ring Detective!")


if __name__ == "__main__":
    main()

