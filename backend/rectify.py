# ---- rectify.py ----

from document_pipeline import DocumentRectifierOCR
from PIL import Image
import numpy as np

# Initialize the rectifier (loads models once)
rectifier = DocumentRectifierOCR()

def rectify_single_image(image_path: str) -> Image.Image:
    """
    Rectify an image given its path and return a PIL Image.
    """
    try:
        scanned_image = rectifier.rectify_image(image_path)
        return scanned_image
    except Exception as e:
        print(f"[Error] Failed to rectify {image_path}: {e}")
        return None
