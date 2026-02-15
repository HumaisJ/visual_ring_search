import streamlit as st
from PIL import Image
import os
import numpy as np
import io

# Make sure engine.py is in the same directory
from engine import RobustJewelrySearchEngine, Config

st.set_page_config(layout="wide", page_title="Visual Ring Search Engine")

@st.cache_resource
def load_engine():
    """Load the search engine only once and cache it."""
    engine = RobustJewelrySearchEngine()
    engine.initialize() # This will now automatically try to load the index
    return engine

def main():
    st.title("💍 Visual Ring Search Engine")
    st.write("A high-accuracy system for finding rings based on design similarity.")

    engine = load_engine()

    # --- Sidebar for Controls ---
    with st.sidebar:
        st.header("Controls")
        
        # --- Catalog Management ---
        st.subheader("Catalog Management")

        # <<< NEW LOGIC: Check if the index file already exists
        index_path = Config.INDEX_SAVE_PATH
        if os.path.exists(index_path):
            st.success(f"✓ Catalog index found and loaded with {len(engine.catalog_index.features)} items.")
            st.info("You can search immediately. Click 'Rebuild' only if you have changed the images in your catalog folder.")
        else:
            st.warning("Catalog index not found. Please build the index before searching.")

        if st.button("Build / Rebuild Catalog Index"):
            with st.spinner("Building index... This can take a few minutes."):
                progress_bar = st.progress(0, text="Starting catalog build...")
                def update_progress(current, total):
                    progress_bar.progress(current / total, text=f"Processing item {current}/{total}")
                
                engine.build_catalog(Config.CATALOG_FOLDER, progress_callback=update_progress)
                st.success(f"Catalog built successfully with {len(engine.catalog_index.features)} items!")
                progress_bar.empty()
                st.rerun() # <<< NEW: Automatically refresh the app state to reflect the new index.
        
        st.divider()

        # --- Search ---
        st.subheader("Search for a Ring")
        uploaded_file = st.file_uploader("Upload a test image...", type=["png", "jpg", "jpeg"])

    # --- Main Area for Display ---
    if uploaded_file is not None:
        st.header("Query Image")
        
        query_image_bytes = uploaded_file.getvalue()
        query_image_pil = Image.open(io.BytesIO(query_image_bytes)).convert("RGB")

        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(query_image_pil, caption="Uploaded Query Image", use_column_width=True)
        with col2:
            # <<< Simplified Check: Just see if the index has items.
            if not engine.catalog_index or not engine.catalog_index.features:
                 st.error("The catalog index is empty. Please build it using the sidebar.")
            else:
                if st.button("✨ Search for Similar Rings", type="primary", use_container_width=True):
                    # ... (The rest of the search logic remains exactly the same)
                    try:
                        with st.spinner("Analyzing image with Ring Detective and searching..."):
                            search_results = engine.search(query_image_bytes, top_k=15)
                        
                        st.header("⚙️ Processing Analysis")
                        meta = search_results['query_metadata']
                        
                        st.info("The image is first passed to the 'Ring Detective'. The detected ring is cropped and then processed.")
                        
                        analysis_cols = st.columns(4)
                        with analysis_cols[0]:
                            st.image(meta['original_image_rgb'], caption="1. Ring Cropped & BG Removed", use_column_width=True)
                        with analysis_cols[1]:
                            st.image(meta['decomposed_masks']['stones'], caption="2. Stone Mask", use_column_width=True)
                        with analysis_cols[2]:
                            st.image(meta['decomposed_masks']['design_pattern'], caption="3. Design Mask", use_column_width=True)
                        with analysis_cols[3]:
                            st.image(search_results['processed_pattern_image'], caption="4. Normalized Design", use_column_width=True)

                        st.divider()
                        st.header("✅ Search Results")

                        for match_type, matches in [("Exact Matches", search_results['exact_matches']), 
                                                    ("Similar Matches", search_results['similar_matches']), 
                                                    ("Other Matches", search_results['other_matches'])]:
                            if matches:
                                st.subheader(match_type)
                                num_cols = 5
                                cols = st.columns(num_cols)
                                for i, res in enumerate(matches):
                                    with cols[i % num_cols]:
                                        img_path = os.path.join(Config.CATALOG_FOLDER, res['metadata']['filename'])
                                        st.image(Image.open(img_path), use_column_width=True)
                                        st.metric("Similarity", f"{res['similarity']*100:.1f}%")
                                        
                    except ValueError as e:
                        if "No ring detected" in str(e):
                            st.error(f"**Search Failed:** The 'Ring Detective' did not find a ring in the uploaded image. Please try another image.")
                        else:
                            st.error(f"An unexpected error occurred during search: {e}")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")
    else:
        st.info("Upload an image in the sidebar to begin your search.")

if __name__ == "__main__":
    main()
