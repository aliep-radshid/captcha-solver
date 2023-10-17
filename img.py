import cv2
import base64
import numpy as np
from glob import glob

templates_5 = map(lambda f5: cv2.imread(
    f5, cv2.IMREAD_GRAYSCALE), glob('./templates/5*.jpg'))


def readb64(encoded_data):
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def find5(img_cv):
    for template in templates_5:
        res = cv2.matchTemplate(img_cv, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.5
        loc = np.where(res >= threshold)
        points = zip(*loc[::-1])
        if len(list(points)) > 0:
            return True
    return False
