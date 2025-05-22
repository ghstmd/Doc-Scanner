import cv2
from image_processing import *

image = cv2.imread("rectified/img1_rec.png")
text = extract_text_from_image(image, lang='vie')
print(text)

processed_image = preprocess_for_ocr(image)
cv2.imshow("Processed Image", processed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()