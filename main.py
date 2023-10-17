import cv2
import pytesseract
from matplotlib import pyplot as plt
import numpy as np

from api import call_captcha
from img import find5

pytesseract.pytesseract.tesseract_cmd = r'E:\Program Files\Tesseract-OCR\tesseract.exe'

captcha_code, img_cv = call_captcha()
# img_cv = img_cv[5:40, 20:130]

img_cv = cv2.copyMakeBorder(img_cv, 25, 25, 25, 25,
                            cv2.BORDER_CONSTANT, value=(255, 255, 255))


# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
# img_cv = cv2.morphologyEx(img_cv, cv2.MORPH_OPEN, kernel)


hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)

mask = cv2.inRange(hsv, (0, 0, 150), (255, 10, 255))

# Build mask of non black pixels.
nzmask = cv2.inRange(hsv, (0, 0, 10), (255, 255, 255))

# Erode the mask - all pixels around a black pixels should not be masked.
nzmask = cv2.erode(nzmask, np.ones((5, 5)))

mask = mask & nzmask

new_img = img_cv.copy()
new_img[np.where(mask)] = 255
img_cv = new_img

for psm in [7, 8, 11]:
    ocr_text = pytesseract.image_to_string(
        img_cv, lang="eng", config=str.format("--psm {0} -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ", psm))
    ocr_text = ocr_text.strip()

    print(psm, ':', ocr_text)

# if find5(img_cv) and ocr_text.find('S') != -1:
#     print('possible 5/S')

# # Set our filtering parameters
# # Initialize parameter setting using cv2.SimpleBlobDetector
# params = cv2.SimpleBlobDetector_Params()

# # Set Area filtering parameters
# params.filterByArea = True
# params.minArea = 100

# # Set Circularity filtering parameters
# params.filterByCircularity = True
# params.minCircularity = 0.9

# # Set Convexity filtering parameters
# params.filterByConvexity = True
# params.minConvexity = 0.2

# # Set inertia filtering parameters
# params.filterByInertia = True
# params.minInertiaRatio = 0.01

# # Create a detector with the parameters
# detector = cv2.SimpleBlobDetector_create(params)

# # Detect blobs
# keypoints = detector.detect(img_cv)

# # Draw blobs on our image as red circles
# blank = np.zeros((1, 1))
# blobs = cv2.drawKeypoints(img_cv, keypoints, blank, (0, 0, 255),
#                           cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


# plt.imshow(mask, 'gray')
# plt.show()
# plt.imshow(nzmask, 'gray')
# plt.show()
# plt.imshow(new_img, 'gray')
# plt.show()
plt.imshow(img_cv, 'gray')
plt.show()
