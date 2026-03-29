import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io


# --- CONFIGURATION ---
# Provide the FULL, ABSOLUTE path to your trained 'best.pt' model file.
# (Right-click on 'best.pt' -> "Copy as path")
MODEL_PATH = "E:\\Punjab University\\MoblieAppDev\\visual_ring_search\\notebooks\\runs\\models\\ring_detector_run\\weights\\best.pt"
# --- END OF CONFIGURATION ---

@st.cache_resource
def load_detector_model():
    """Loads the YOLO model once and caches it."""
    try:
        model = YOLO(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}. Please check the MODEL_PATH in the script.")
        return None

st.set_page_config(layout="wide", page_title="Ring Detector Tester")
st.title("🕵️‍♂️ Ring Detective: Standalone Tester")
st.write("Upload an image to see if the YOLOv8 model can find a ring.")

model = load_detector_model()

if model:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        col1, col2 = st.columns(2)

        # Read and display the original image
        image = Image.open(uploaded_file).convert("RGB")
        with col1:
            st.subheader("Original Image")
            st.image(image,  width="stretch")

        with col2:
            st.subheader("Detection Result")
            with st.spinner("Running detection..."):
                # Run prediction
                results = model.predict(image, verbose=False)
                result = results[0] # Get the first result

                if len(result.boxes) == 0:
                    st.warning("No ring was detected in this image.")
                else:
                    st.success(f"Found {len(result.boxes)} ring(s)!")
                    
                    # Use the built-in plot function from YOLO to draw boxes
                    result_image = result.plot()
                    
                    # Convert color from BGR (OpenCV format) to RGB (PIL/Streamlit format)
                    result_image_rgb = Image.fromarray(result_image[:, :, ::-1])
                    
                    st.image(result_image_rgb, caption="Image with bounding boxes",  width="stretch")

                    # Print details for each detected box
                    for box in result.boxes:
                        confidence = box.conf.item()
                        st.write(f" - Found a 'ring' with **{confidence:.1%}** confidence.")
