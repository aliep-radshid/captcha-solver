import requests
import json

from img import readb64


def call_captcha():
    r = requests.post(
        "https://report.sipaad.ir/IGS.ITM.WebSiteServiceMiddle/WebServices/Authentication.asmx/GetCaptcha", json={"callback": ""})

    captcha_code, captcha_img = json.loads(
        r.text[0:r.text.index('{"d":null}')])

    return captcha_code, readb64(captcha_img)
