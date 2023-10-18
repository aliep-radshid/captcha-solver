import cv2
import requests
import json

from img import readb64


def _strip_dnull(text: str):
    return text[0:text.index('{"d":null}')]


def call_captcha(mode=cv2.IMREAD_COLOR):
    r = requests.post(
        "https://report.sipaad.ir/IGS.ITM.WebSiteServiceMiddle/WebServices/Authentication.asmx/GetCaptcha", json={"callback": ""})

    captcha_code, captcha_img = json.loads(
        r.text[0:r.text.index('{"d":null}')])

    return captcha_code, readb64(captcha_img, mode)


def call_get_record(token: str, companycode: int, IMEI: str):
    body = {"token": token, "callback": "", "companycode": companycode, "IMEI": IMEI,
            "VIN": "", "PlaqueType": "", "PlaqueSN": "", "PlaqueID": "", "DeviceType": 1}
    r = requests.post(
        "https://report.sipaad.ir/IGS.ITM.WebSiteServiceMiddle/WebServices/ReportsTest.asmx/GetRecord", json=body)
