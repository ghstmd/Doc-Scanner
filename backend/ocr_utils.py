# ocr_utils.py
import pytesseract
from PIL import Image

def perform_ocr(image_path):
    img = Image.open(image_path)
    
    # Get OCR data with confidence per word
    ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

    text = []
    confidences = []
    
    for i in range(len(ocr_data['text'])):
        word = ocr_data['text'][i].strip()
        conf = int(ocr_data['conf'][i])
        if word and conf > 0:
            text.append(word)
            confidences.append(conf)
    
    full_text = ' '.join(text)
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0

    return full_text, avg_confidence
