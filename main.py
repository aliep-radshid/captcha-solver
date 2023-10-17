import cv2
import pytesseract
from matplotlib import pyplot as plt
# import numpy as np
from glob import glob
from strsimpy.normalized_levenshtein import NormalizedLevenshtein

normalized_levenshtein = NormalizedLevenshtein()
pytesseract.pytesseract.tesseract_cmd = r'E:\Program Files\Tesseract-OCR\tesseract.exe'

for f in glob('images/*.jpg'):
    actual_text = f.split('\\')[1].split('.jpg')[0]
    img_cv = cv2.imread(f, cv2.IMREAD_GRAYSCALE)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    img_cv = cv2.morphologyEx(img_cv, cv2.MORPH_OPEN, kernel)

    # Page segmentation modes:
    #     0    Orientation and script detection (OSD) only.
    #     1    Automatic page segmentation with OSD.
    #     2    Automatic page segmentation, but no OSD, or OCR. (not implemented)
    #     3    Fully automatic page segmentation, but no OSD. (Default)
    #     4    Assume a single column of text of variable sizes.
    #     5    Assume a single uniform block of vertically aligned text.
    #     6    Assume a single uniform block of text.
    #     7    Treat the image as a single text line.
    #     8    Treat the image as a single word.
    #     9    Treat the image as a single word in a circle.
    #     10    Treat the image as a single character.
    #     11    Sparse text. Find as much text as possible in no particular order.
    #     12    Sparse text with OSD.
    #     13    Raw line. Treat the image as a single text line,
    #         bypassing hacks that are Tesseract-specific.

    ocr_text = pytesseract.image_to_string(
        img_cv, lang="enm", config="--psm 12 --dpi 96 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    ocr_text = ocr_text.strip()

    distance = round(normalized_levenshtein.similarity(
        actual_text, ocr_text), 1)

    print(actual_text, ocr_text, distance)
