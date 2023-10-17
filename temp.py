import cv2
import pytesseract
from matplotlib import pyplot as plt
import numpy as np
from glob import glob
from strsimpy.normalized_levenshtein import NormalizedLevenshtein

normalized_levenshtein = NormalizedLevenshtein()
pytesseract.pytesseract.tesseract_cmd = r'E:\Program Files\Tesseract-OCR\tesseract.exe'

for f in glob('images/*5.jpg'):
    actual_text = f.split('\\')[1].split('.jpg')[0]
    img_cv = cv2.imread(f, cv2.IMREAD_GRAYSCALE)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    img_cv = cv2.morphologyEx(img_cv, cv2.MORPH_OPEN, kernel)

    for f5 in glob('./templates/5*.jpg'):
        template = cv2.imread(f5, cv2.IMREAD_GRAYSCALE)
        w, h = template.shape[::-1]
        res = cv2.matchTemplate(img_cv, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.5
        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):
            cv2.rectangle(img_cv, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 1)

    plt.imshow(img_cv, 'gray')
    plt.show()
