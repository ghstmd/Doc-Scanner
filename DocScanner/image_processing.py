from model import DocScanner
from seg import U2NETP

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
from PIL import Image
import argparse

import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.msk = U2NETP(3, 1).to(device)
        self.bm = DocScanner().to(device)


    def forward(self, x):
        msk, _1,_2,_3,_4,_5,_6 = self.msk(x)
        msk = (msk > 0.5).float()
        x = msk * x

        bm = self.bm(x, iters=12, test_mode=True)
        bm = (2 * (bm / 286.8) - 1) * 0.99

        return bm


def reload_seg_model(model, path=""):
    if not bool(path):
        return model
    else:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path, map_location=device)

        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if k[6:] in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        return model


def reload_rec_model(model, path=""):
    if not bool(path):
        return model
    else:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path, map_location=device)

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        return model


def rec(seg_model_path, rec_model_path, distorrted_path, save_path):
    # distorted images list
    img_list = os.listdir(distorrted_path)

    # creat save path for rectified images
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # net init
    net = Net().to(device)

    # reload seg model
    reload_seg_model(net.msk, seg_model_path)
    # reload rec model
    reload_rec_model(net.bm, rec_model_path)

    net.eval()

    for img_path in img_list:
        name = img_path.split('.')[-2]  # image name
        img_path = distorrted_path + img_path  # image path

        im_ori = np.array(Image.open(img_path))[:, :, :3] / 255.
        h, w, _ = im_ori.shape
        im = cv2.resize(im_ori, (288, 288))
        im = im.transpose(2, 0, 1)
        im = torch.from_numpy(im).float().unsqueeze(0)

        with torch.no_grad():
            bm = net(im.to(device))
            bm = bm.cpu()

            # save rectified image
            bm0 = cv2.resize(bm[0, 0].numpy(), (w, h))  # x flow
            bm1 = cv2.resize(bm[0, 1].numpy(), (w, h))  # y flow
            bm0 = cv2.blur(bm0, (3, 3))
            bm1 = cv2.blur(bm1, (3, 3))
            lbl = torch.from_numpy(np.stack([bm0, bm1], axis=2)).unsqueeze(0)  # h * w * 2
            out = F.grid_sample(torch.from_numpy(im_ori).permute(2, 0, 1).unsqueeze(0).float(), lbl, align_corners=True)
            cv2.imwrite(save_path + name + '_rec' + '.png', (((out[0]*255).permute(1, 2, 0).numpy())[:,:,::-1]).astype(np.uint8))


def rectify_images():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seg_model_path', default='./model_pretrained/seg.pth')
    parser.add_argument('--rec_model_path', default='./model_pretrained/DocScanner-L.pth')
    parser.add_argument('--distorrted_path', default='./distorted/')
    parser.add_argument('--rectified_path', default='./rectified/')
    opt = parser.parse_args()

    rec(seg_model_path=opt.seg_model_path,
        rec_model_path=opt.rec_model_path,
        distorrted_path=opt.distorrted_path,
        save_path=opt.rectified_path)
    

def rectify_single_image(img_path: str):
    """
    Takes paths to the segmentation & rectification model weights and
    a single distorted image file, returns the rectified PIL Image.
    """
    seg_model_path="./model_pretrained/seg.pth"
    rec_model_path="./model_pretrained/DocScanner-L.pth"
    # Initialize and load models
    net = Net().to(device)
    reload_seg_model(net.msk, seg_model_path)
    reload_rec_model(net.bm, rec_model_path)
    net.eval()

    # Load and preprocess
    im_ori = np.array(Image.open(img_path))[:, :, :3] / 255.0
    h, w, _ = im_ori.shape
    im = cv2.resize(im_ori, (288, 288)).transpose(2, 0, 1)
    im = torch.from_numpy(im).float().unsqueeze(0)

    with torch.no_grad():
        # Forward pass
        bm = net(im.to(device)).cpu()
        # Resize and smooth flows
        flow_x = cv2.blur(cv2.resize(bm[0, 0].numpy(), (w, h)), (3, 3))
        flow_y = cv2.blur(cv2.resize(bm[0, 1].numpy(), (w, h)), (3, 3))
        # Create sampling grid
        grid = torch.from_numpy(np.stack([flow_x, flow_y], axis=2)).unsqueeze(0)
        # Warp original
        tensor_ori = torch.from_numpy(im_ori).permute(2, 0, 1).unsqueeze(0).float()
        out = F.grid_sample(tensor_ori, grid, align_corners=True)
        # Convert back to HxWxBGR uint8
        out_np = (((out[0] * 255).permute(1, 2, 0).numpy())[:, :, ::-1]).astype(np.uint8)

    # Return a PIL Image for easy further use
    return Image.fromarray(out_np)

import cv2
import numpy as np
import pytesseract

def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    Loại bỏ bóng, tăng tương phản, giữ nét chữ và loại bỏ nhiễu gần chữ.
    """
    # 1. Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Shadow removal via background subtraction
    dilated = cv2.dilate(gray, np.ones((7, 7), np.uint8))
    bg = cv2.medianBlur(dilated, 21)
    diff = 255 - cv2.absdiff(gray, bg)
    norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)

    # 3. Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(norm)

    # 4. Denoising (bilateral filter keeps edges sharp)
    denoised = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)

    # 5. Sharpening
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(denoised, -1, sharpen_kernel)

    # 6. Adaptive thresholding
    binary = cv2.adaptiveThreshold(
        sharpened,
        255,  # max value
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=13, # # smaller block size for better detail
        C=24  # slightly lower to reduce bleed
    )

    # 7. Morphological opening to remove small noise
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return cleaned



def extract_text_from_image(image: np.ndarray, lang: str = 'vie') -> str:
    """
    Tiền xử lý ảnh và trích xuất văn bản bằng pytesseract.
    lang: 'eng' cho tiếng Anh, 'vie' cho tiếng Việt, hoặc 'eng+vie'
    """
    preprocessed = preprocess_for_ocr(image)
    text = pytesseract.image_to_string(preprocessed, lang=lang)
    return text



# rectified_img = rectify_single_image(
#     './distorted/img1.jpg'
# )
# rectified_img.save('rectified/img1_rec.png')