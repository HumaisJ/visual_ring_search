from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

# --- CONFIGURATION: YOU MUST CHANGE THESE TWO PATHS ---

# 1. Provide the FULL, ABSOLUTE path to your trained 'best.pt' model file.
#    (Right-click on 'best.pt' -> "Copy as path")
MODEL_PATH = "E:\\Punjab University\\MoblieAppDev\\visual_ring_search\\notebooks\\runs\\models\\ring_detector_run\\weights\\best.pt"

# 2. Provide the FULL, ABSOLUTE path to an image you want to test.
#    Use a messy image from your '01_raw' dataset for a good test.
TEST_IMAGE_PATH = "E:\\Punjab University\\MoblieAppDev\\visual_ring_search\\dataset\\01_raw\\photo-1605089315599-ca966e96b56a.jpg"

# --- END OF CONFIGURATION ---


def main():
    """
    Loads the YOLO Ring Detective, runs it on a single image,
    and displays the result with bounding boxes.
    """
    print(f"Loading model from: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"ERROR: Could not load the model. Please check the MODEL_PATH. Error: {e}")
        return

    print(f"Loading image from: {TEST_IMAGE_PATH}")
    try:
        image = Image.open(TEST_IMAGE_PATH).convert("RGB")
    except FileNotFoundError:
        print(f"ERROR: Could not find the image. Please check the TEST_IMAGE_PATH.")
        return
    
    print("Running detection...")
    results = model.predict(image, verbose=False)
    
    # Get the first result object
    result = results[0]

    # Convert PIL Image to an OpenCV image (for drawing)
    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    if len(result.boxes) == 0:
        print("\n--- RESULT: NO RING DETECTED ---")
    else:
        print(f"\n--- RESULT: {len(result.boxes)} RING(S) DETECTED! ---")
        for box in result.boxes:
            # Get coordinates, confidence, and class
            x1, y1, x2, y2 = [int(c) for c in box.xyxy[0]]
            confidence = box.conf.item()
            class_id = int(box.cls.item())
            label = model.names[class_id]
            
            print(f"  - Found '{label}' with {confidence:.2%} confidence.")
            
            # Draw the bounding box
            cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Create the label text
            text = f"{label}: {confidence:.2%}"
            
            # Put the label text above the box
            cv2.putText(image_cv, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the image in a window
    cv2.imshow("Ring Detection Result", image_cv)
    print("\nDisplaying result image. Press any key in the image window to close.")
    cv2.waitKey(0)  # Wait for a key press
    cv2.destroyAllWindows() # Close the window

if __name__ == "__main__":
    main()
