import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import (
    preprocess_input as effnet_preprocess,
)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
import os
from PIL import Image
import logging
import time
from rembg import remove
import io
import pickle
from sklearn.neighbors import NearestNeighbors
from ultralytics import YOLO  # <<< NEW: Import the YOLO library

# ============= CONFIGURATION =============
class Config:
    # <<< NEW: Path to our trained Ring Detective model
    # This relative path works because app.py and engine.py are in the 'app' folder
    DETECTOR_MODEL_PATH = "E:\\Punjab University\\MoblieAppDev\\visual_ring_search\\notebooks\\runs\\models\\ring_detector_run\\weights\\best.pt"
    DETECTOR_CONFIDENCE_THRESHOLD = 0.5 # <<< NEW: Minimum confidence to accept a detection

    CATALOG_FOLDER = "../dataset/03_design_catalog/" # <<< UPDATED: Point to our new data structure
    INDEX_SAVE_PATH = "../models/catalog_index_robust.pkl" # <<< UPDATED: Point to the models folder
    
    TARGET_SIZE = (224, 224)
    USE_TTA = True
    TTA_ROTATIONS = [0, 90, 180, 270]
    TTA_FLIPS = [False, True]
    EXACT_MATCH_THRESHOLD = 0.98
    SIMILAR_MATCH_THRESHOLD = 0.75
    USE_GPU = True
    INDEX_ALGORITHM = "brute"
    USE_PERCEPTUAL_HASH = True
    USE_MEDIAN_AVERAGING = True

# (Logging and GPU configuration remain the same)
# ============= LOGGING & GPU =============
def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger(__name__)

logger = setup_logging()

def configure_gpu():
    if Config.USE_GPU:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"✓ GPU configured: {len(gpus)} device(s) available")
            except RuntimeError as e:
                logger.warning(f"GPU configuration error: {e}")
        else:
            logger.warning("No GPU found, using CPU")

# (PerceptualHash, RingDecomposer, and DesignNormalizer classes remain the same)
# ============= PERCEPTUAL HASHING =============
class PerceptualHash:
    @staticmethod
    def dhash(image, hash_size=16):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        resized = cv2.resize(gray, (hash_size + 1, hash_size))
        diff = resized[:, 1:] > resized[:, :-1]
        return diff.flatten()

    @staticmethod
    def hamming_distance(hash1, hash2):
        return np.sum(hash1 != hash2)

    @staticmethod
    def hash_similarity(hash1, hash2):
        dist = PerceptualHash.hamming_distance(hash1, hash2)
        return 1 - (dist / len(hash1))

# ============= STAGE 1: VISUAL DECOMPOSITION =============
class RingDecomposer:
    def _segment_stones(self, image_rgb):
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        sat_mask = cv2.threshold(s, 100, 255, cv2.THRESH_BINARY)[1]
        val_mask = cv2.threshold(v, 180, 255, cv2.THRESH_BINARY)[1]
        stone_mask = cv2.bitwise_and(sat_mask, val_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        stone_mask = cv2.morphologyEx(stone_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return stone_mask

    def _segment_design_pattern(self, image_rgb, foreground_mask):
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        smooth_base = cv2.GaussianBlur(gray, (31, 31), 0)
        design_retained = cv2.GaussianBlur(gray, (5, 5), 0)
        design_texture = cv2.absdiff(design_retained, smooth_base)
        _, design_mask = cv2.threshold(design_texture, 5, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        design_mask = cv2.morphologyEx(design_mask, cv2.MORPH_CLOSE, kernel)
        design_mask = cv2.dilate(design_mask, kernel, iterations=1)
        design_mask = cv2.bitwise_and(design_mask, foreground_mask)
        return design_mask

    def decompose(self, image_rgb, alpha_mask):
        foreground_mask = alpha_mask
        stone_mask = self._segment_stones(image_rgb)
        stone_mask = cv2.bitwise_and(stone_mask, foreground_mask)
        design_mask = self._segment_design_pattern(image_rgb, foreground_mask)
        design_mask = cv2.subtract(design_mask, stone_mask)
        metal_base_mask = cv2.subtract(foreground_mask, stone_mask)
        metal_base_mask = cv2.subtract(metal_base_mask, design_mask)
        return {"metal_base": metal_base_mask, "stones": stone_mask, "design_pattern": design_mask}

# ============= STAGE 2: DESIGN NORMALIZATION =============
class DesignNormalizer:
    def normalize(self, image_rgb, design_mask):
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        design_texture = cv2.bitwise_and(gray, gray, mask=design_mask)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        normalized_design = clahe.apply(design_texture)
        normalized_design = cv2.bitwise_and(normalized_design, normalized_design, mask=design_mask)
        _, final_design_descriptor = cv2.threshold(normalized_design, 50, 255, cv2.THRESH_BINARY)
        final_design_descriptor_rgb = cv2.cvtColor(final_design_descriptor, cv2.COLOR_GRAY2RGB)
        return final_design_descriptor_rgb

# ============= MAIN PREPROCESSING PIPELINE =============
class RobustImagePreprocessor:
    def __init__(self, config=Config):
        self.config = config
        self.phash = PerceptualHash()
        self.decomposer = RingDecomposer()
        self.normalizer = DesignNormalizer()
        # <<< NEW: Load our trained Ring Detective model when the preprocessor is created
        try:
            self.detector_model = YOLO(self.config.DETECTOR_MODEL_PATH)
            logger.info("✓ Ring Detector model loaded successfully.")
        except Exception as e:
            logger.error(f"FATAL: Could not load Ring Detector model. Ensure the path is correct: {self.config.DETECTOR_MODEL_PATH}. Error: {e}")
            self.detector_model = None

    # <<< NEW: This entirely new function uses YOLO to find and crop the ring
    def detect_and_crop_ring(self, image_rgb):
        if self.detector_model is None:
            raise RuntimeError("Ring detector model is not loaded.")

        results = self.detector_model.predict(image_rgb, verbose=False)
        result = results[0] # Get results for the first image

        if len(result.boxes) == 0:
            return None # No rings detected

        # Get the box with the highest confidence
        best_box = max(result.boxes, key=lambda box: box.conf.item())
        
        if best_box.conf.item() < self.config.DETECTOR_CONFIDENCE_THRESHOLD:
            logger.warning(f"Ring detected but confidence ({best_box.conf.item():.2f}) is below threshold ({self.config.DETECTOR_CONFIDENCE_THRESHOLD}).")
            return None

        # Extract coordinates and crop
        x1, y1, x2, y2 = [int(coord) for coord in best_box.xyxy[0]]
        cropped_ring = image_rgb[y1:y2, x1:x2]
        
        logger.info(f"Ring detected with confidence {best_box.conf.item():.2f}. Cropping.")
        return cropped_ring

    def remove_background(self, image_bytes):
        # (This function remains the same)
        try:
            img_bg_removed_bytes = remove(image_bytes)
            img_bg_removed_pil = Image.open(io.BytesIO(img_bg_removed_bytes)).convert("RGBA")
            img_bg_removed_np = np.array(img_bg_removed_pil)
            img_rgb = cv2.cvtColor(img_bg_removed_np, cv2.COLOR_RGBA2RGB)
            alpha_mask = img_bg_removed_np[:, :, 3]
            return img_rgb, alpha_mask
        except Exception as e:
            logger.error(f"Background removal failed: {e}")
            return None, None

    # <<< UPDATED: The main processing logic is heavily modified
    def process_image(self, image_path_or_bytes):
        start_time = time.time()
        metadata = {}
        
        try:
            # --- Load Image ---
            if isinstance(image_path_or_bytes, str):
                metadata["path"] = image_path_or_bytes
                img_pil = Image.open(image_path_or_bytes).convert("RGB")
                full_image_rgb = np.array(img_pil)
            else: # Handle bytes for uploaded files
                metadata["path"] = "uploaded_image"
                img_pil = Image.open(io.BytesIO(image_path_or_bytes)).convert("RGB")
                full_image_rgb = np.array(img_pil)

            # --- 1. DETECT AND CROP RING (New First Step) ---
            cropped_ring_rgb = self.detect_and_crop_ring(full_image_rgb)
            if cropped_ring_rgb is None:
                raise ValueError("No ring detected with sufficient confidence.")
            
            # Convert cropped numpy array back to bytes for background removal
            cropped_pil = Image.fromarray(cropped_ring_rgb)
            buffer = io.BytesIO()
            cropped_pil.save(buffer, format="PNG")
            cropped_bytes = buffer.getvalue()

            # --- 2. REMOVE BACKGROUND (Now runs on the cropped image) ---
            original_rgb, alpha_mask = self.remove_background(cropped_bytes)
            if alpha_mask is None:
                raise IOError("Background removal failed on cropped image.")

            # --- 3. DECOMPOSE AND NORMALIZE DESIGN ---
            denoised_rgb = cv2.bilateralFilter(original_rgb, 9, 75, 75)
            decomposed_masks = self.decomposer.decompose(denoised_rgb, alpha_mask)
            normalized_design_pattern = self.normalizer.normalize(denoised_rgb, decomposed_masks['design_pattern'])
            
            # --- 4. Finalize ---
            img_final = cv2.resize(normalized_design_pattern, self.config.TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
            phash_val = self.phash.dhash(img_final) if self.config.USE_PERCEPTUAL_HASH else None
            
            metadata['processing_time'] = time.time() - start_time
            metadata['perceptual_hash'] = phash_val
            metadata['decomposed_masks'] = decomposed_masks
            metadata['original_image_rgb'] = original_rgb # This is now the cropped, background-removed image

            logger.info(f"Processed {metadata['path']}: {metadata['processing_time']:.3f}s")
            return img_final, metadata
        except Exception as e:
            logger.error(f"Error processing {metadata.get('path', 'image')}: {e}")
            raise

# (TTAFeatureExtractor, CatalogIndex, and RobustJewelrySearchEngine classes remain the same, but will now use the updated preprocessor)
# ============= FEATURE EXTRACTOR =============
class TTAFeatureExtractor:
    def __init__(self, model_type='efficientnet', config=Config):
        self.config = config
        base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(*Config.TARGET_SIZE, 3))
        self.preprocess_fn = effnet_preprocess
        x = GlobalAveragePooling2D()(base.output)
        self.model = Model(inputs=base.input, outputs=x)
        logger.info(f"TTA Feature extractor initialized: {model_type}")

    def apply_tta_transform(self, img, rotation=0, flip=False):
        h, w = img.shape[:2]
        if rotation != 0:
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, rotation, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=[0,0,0])
        if flip:
            img = cv2.flip(img, 1)
        return img

    def extract_features_single(self, img, normalize=True):
        img_array = np.expand_dims(img, axis=0)
        img_array = self.preprocess_fn(img_array.astype('float32'))
        features = self.model.predict(img_array, verbose=0).flatten()
        if normalize:
            norm = np.linalg.norm(features)
            if norm > 0: features /= norm
        return features

    def extract_features(self, img, use_tta=True):
        if not use_tta:
            return self.extract_features_single(img)
        
        all_features = []
        for rotation in self.config.TTA_ROTATIONS:
            for flip in self.config.TTA_FLIPS:
                img_transformed = self.apply_tta_transform(img, rotation, flip)
                features = self.extract_features_single(img_transformed, normalize=False)
                all_features.append(features)

        features_agg = np.median(all_features, axis=0) if self.config.USE_MEDIAN_AVERAGING else np.mean(all_features, axis=0)
        norm = np.linalg.norm(features_agg)
        if norm > 0: features_agg /= norm
        return features_agg

# ============= CATALOG & SEARCH =============
class CatalogIndex:
    def __init__(self, algorithm='brute', metric='cosine', config=Config):
        self.config = config
        self.features = []
        self.metadata = []
        self.perceptual_hashes = []
        self.index = None
        self.phash = PerceptualHash()

    def add_item(self, features, metadata):
        self.features.append(features)
        self.metadata.append(metadata)
        if self.config.USE_PERCEPTUAL_HASH:
            self.perceptual_hashes.append(metadata.get('perceptual_hash'))

    def build_index(self):
        if not self.features: return
        features_array = np.array(self.features)
        self.index = NearestNeighbors(n_neighbors=min(50, len(self.features)), algorithm=self.config.INDEX_ALGORITHM, metric='cosine').fit(features_array)

    def search(self, query_features, query_phash=None, top_k=10):
        if self.index is None: raise ValueError("Index not built.")
        
        query_features = np.array(query_features).reshape(1, -1)
        distances, indices = self.index.kneighbors(query_features, n_neighbors=top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            similarity = 1 - dist
            if self.config.USE_PERCEPTUAL_HASH and query_phash is not None and self.perceptual_hashes[idx] is not None:
                phash_sim = self.phash.hash_similarity(query_phash, self.perceptual_hashes[idx])
                if phash_sim > 0.98: similarity = max(similarity, phash_sim)
            
            results.append({'metadata': self.metadata[idx], 'similarity': float(np.clip(similarity, 0.0, 1.0))})

        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'features': self.features, 'metadata': self.metadata, 'perceptual_hashes': self.perceptual_hashes}, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.features = data['features']
        self.metadata = data['metadata']
        self.perceptual_hashes = data.get('perceptual_hashes', [])
        self.build_index()

# ============= MAIN SEARCH ENGINE FACADE =============
class RobustJewelrySearchEngine:
    def __init__(self, config=Config):
        self.config = config
        self.preprocessor = RobustImagePreprocessor(config)
        self.feature_extractor = None
        self.catalog_index = None

    def initialize(self, model_type='efficientnet'):
        configure_gpu()
        self.feature_extractor = TTAFeatureExtractor(model_type, self.config)
        self.catalog_index = CatalogIndex(config=self.config)
        
        if os.path.exists(self.config.INDEX_SAVE_PATH):
            logger.info("Loading existing catalog...")
            self.load_catalog()
        else:
            logger.info("No existing catalog found. Please build it.")

    def build_catalog(self, catalog_folder, progress_callback=None):
        logger.info("Building new catalog...")
        self.catalog_index = CatalogIndex(config=self.config) # Reset index
        image_files = [f for f in os.listdir(catalog_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for idx, filename in enumerate(image_files):
            img_path = os.path.join(catalog_folder, filename)
            try:
                processed_pattern, metadata = self.preprocessor.process_image(img_path)
                features = self.feature_extractor.extract_features(processed_pattern, use_tta=self.config.USE_TTA)
                metadata['filename'] = filename
                # Don't store large image data in the index
                metadata.pop('original_image_rgb', None)
                metadata.pop('decomposed_masks', None)
                self.catalog_index.add_item(features, metadata)
                if progress_callback:
                    progress_callback(idx + 1, len(image_files))
            except Exception as e:
                logger.error(f"Failed to process {filename}: {e}")

        self.catalog_index.build_index()
        self.catalog_index.save(self.config.INDEX_SAVE_PATH)
        logger.info(f"✓ Catalog built: {len(self.catalog_index.features)} items")

    def load_catalog(self):
        self.catalog_index.load(self.config.INDEX_SAVE_PATH)

    def search(self, query_image_bytes, top_k=10):
        processed_pattern, query_metadata = self.preprocessor.process_image(query_image_bytes)
        query_features = self.feature_extractor.extract_features(processed_pattern, use_tta=self.config.USE_TTA)
        query_phash = query_metadata.get('perceptual_hash')

        results = self.catalog_index.search(query_features, query_phash, top_k)

        exact_matches, similar_matches, other_matches = [], [], []
        for r in results:
            if r['similarity'] >= self.config.EXACT_MATCH_THRESHOLD: exact_matches.append(r)
            elif r['similarity'] >= self.config.SIMILAR_MATCH_THRESHOLD: similar_matches.append(r)
            else: other_matches.append(r)

        return {
            'query_metadata': query_metadata,
            'processed_pattern_image': processed_pattern,
            'exact_matches': exact_matches,
            'similar_matches': similar_matches,
            'other_matches': other_matches,
        }
