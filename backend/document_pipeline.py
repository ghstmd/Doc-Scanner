# ---- document_pipeline.py ----

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from model import DocScanner
from seg import U2NETP
import pytesseract
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DocumentRectifierOCR:
    def __init__(self,
                 seg_model_path='backend/model_pretrained/seg.pth',
                 rec_model_path='backend/model_pretrained/DocScanner-L.pth'):
        # Initialize network
        self.net = self._build_model()
        self._load_models(seg_model_path, rec_model_path)
        self.net.eval()

    def _build_model(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.msk = U2NETP(3, 1).to(device)
                self.bm = DocScanner().to(device)

            def forward(self, x):
                msk, *_ = self.msk(x)
                msk = (msk > 0.5).float()
                x = msk * x
                bm = self.bm(x, iters=12, test_mode=True)
                bm = (2 * (bm / 286.8) - 1) * 0.99
                return bm
        return Net().to(device)

    def _load_models(self, seg_path, rec_path):
        try:
            seg_model_dict = self.net.msk.state_dict()
            seg_pretrained_dict = torch.load(seg_path, map_location=device)
            seg_pretrained_dict = {k[6:]: v for k, v in seg_pretrained_dict.items() if k[6:] in seg_model_dict}
            seg_model_dict.update(seg_pretrained_dict)
            self.net.msk.load_state_dict(seg_model_dict)

            rec_model_dict = self.net.bm.state_dict()
            rec_pretrained_dict = torch.load(rec_path, map_location=device)
            rec_pretrained_dict = {k: v for k, v in rec_pretrained_dict.items() if k in rec_model_dict}
            rec_model_dict.update(rec_pretrained_dict)
            self.net.bm.load_state_dict(rec_model_dict)

            print("Models loaded successfully.")
        except Exception as e:
            print(f"Error loading models: {e}")
            raise


    def rectify_image(self, img_path: str) -> Image.Image:
        """
        Rectify a single image and return as PIL Image
        """
        im_ori = np.array(Image.open(img_path))[:, :, :3] / 255.0
        h, w, _ = im_ori.shape
        im = cv2.resize(im_ori, (288, 288)).transpose(2, 0, 1)
        im = torch.from_numpy(im).float().unsqueeze(0)

        with torch.no_grad():
            bm = self.net(im.to(device)).cpu()
            flow_x = cv2.blur(cv2.resize(bm[0, 0].numpy(), (w, h)), (3, 3))
            flow_y = cv2.blur(cv2.resize(bm[0, 1].numpy(), (w, h)), (3, 3))
            grid = torch.from_numpy(np.stack([flow_x, flow_y], axis=2)).unsqueeze(0)
            tensor_ori = torch.from_numpy(im_ori).permute(2, 0, 1).unsqueeze(0).float()
            out = F.grid_sample(tensor_ori, grid, align_corners=True)
            out_np = (((out[0] * 255).permute(1, 2, 0).numpy())[:, :, ::-1]).astype(np.uint8)

        return Image.fromarray(out_np)

    def preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dilated = cv2.dilate(gray, np.ones((7, 7), np.uint8))
        bg = cv2.medianBlur(dilated, 21)
        diff = 255 - cv2.absdiff(gray, bg)
        norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(norm)
        denoised = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)
        sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(denoised, -1, sharpen_kernel)
        binary = cv2.adaptiveThreshold(
            sharpened, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=13, C=24
        )
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        return cleaned

    def extract_text(self, image: np.ndarray, lang='vie') -> (str, float):
        preprocessed = self.preprocess_for_ocr(image)
        data = pytesseract.image_to_data(preprocessed, lang=lang, output_type=pytesseract.Output.DICT)
        
        texts = []
        confidences = []

        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 0:
                texts.append(data['text'][i])
                confidences.append(float(data['conf'][i]))

        final_text = ' '.join(texts)
        avg_confidence = np.mean(confidences) / 100 if confidences else 0.0

        return final_text, avg_confidence

