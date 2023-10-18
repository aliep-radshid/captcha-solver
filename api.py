import cv2
import requests
import json

from img import readb64


class IGSException(Exception):
    def __init__(self, body: dict):
        self.code = body["ErrCode"]
        super().__init__(body["ErrDesc"])


class IGSLoginInfo():
    """
    {
        "TokenCode": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJhY3RvcnQiOiIxMDI2MDU0NDUyMCIsImdyb3Vwc2lkIjoiMSIsImNlcnRzZXJpYWxudW1iZXIiOiIxNzIuMTYuNDIuMTE2IiwiYXV0aG1ldGhvZCI6IjMiLCJpc3MiOiJpZ3MiLCJhdWQiOiJodHRwOi8vd3d3Lmlnc2l0LmNvbSIsImV4cCI6MTY5NzYzNzYwNiwibmJmIjoxNjk3NjE2MDA2fQ.xkPQNO3AiAfOPAaOHR3DxbyaEvTtYQlqX5xsdbBcehM",
        "Name": "مجموعه مهندسي رادشيد",
        "CurrentDate": "چهارشنبه 1402/07/26",
        "AccessReports": "|DC1|DC2|DC3|RC1|RC2|Exp|",
        "CompanyID": 1
    }
    """

    def __init__(self, body: dict):
        self.token = body["TokenCode"]
        self.name = body["Name"]
        self.company_id = body["CompanyID"]
        self.access_reports = body["AccessReports"]


def _strip_dnull(text: str):
    return text[0:text.index('{"d":null}')]


def _handle_err(body: dict):
    if "ErrCode" in body:
        raise IGSException(body)


def call_captcha(mode=cv2.IMREAD_COLOR):
    r = requests.post(
        "https://report.sipaad.ir/IGS.ITM.WebSiteServiceMiddle/WebServices/Authentication.asmx/GetCaptcha", json={"callback": ""})

    captcha_code, captcha_img = json.loads(_strip_dnull(r.text))

    return captcha_code, readb64(captcha_img, mode)


def call_get_record(token: str, companycode: int, IMEI: str):
    body = {"token": token, "callback": "", "companycode": companycode, "IMEI": IMEI,
            "VIN": "", "PlaqueType": "", "PlaqueSN": "", "PlaqueID": "", "DeviceType": 1}
    r = requests.post(
        "https://report.sipaad.ir/IGS.ITM.WebSiteServiceMiddle/WebServices/ReportsTest.asmx/GetRecord", json=body)
    body = json.loads(_strip_dnull(r.text))
    _handle_err(body)
    return body
    """
    [
        {
            "DEVICEID": 22646.0,
            "DEVICECOMPANYID": 1.0,
            "DEVICECODE": "866897051642090",
            "FREIGHTERID": 1811316.0,
            "DEVICEINSTALLDATE": "1401/08/21",
            "DEVICEINSTALLPLACECITYID": 391.0,
            "DEVICEINSTALLPLACEPROVINCEID": null,
            "DEVICESTATE": 1,
            "DEVICETHUMBPRINT": "83D754E05F242B08AC79D1C278D5E32CA0E4A17C",
            "DEVICEPUBLICKEY": "-----BEGIN CERTIFICATE-----\r\nMIIB3jCCAUegAwIBAgIIfoaa/9FZOOEwDQYJKoZIhvcNAQELBQAwMzELMAkGA1UEBhMCTkwxDDAK\r\nBgNVBAoMA0lHUzEWMBQGA1UEAwwNZ3BzLnNpcGFhZC5pcjAeFw0yMjExMDEwMDAwMDBaFw0zMjEx\r\nMDEwMDAwMDBaMDAxCzAJBgNVBAYTAk5MMQwwCgYDVQQKDANJR1MxEzARBgNVBAMMCklHUyBDbGll\r\nbnQwgZ8wDQYJKoZIhvcNAQEBBQADgY0AMIGJAoGBAJ5DjgL9Q1MffP2D/3k0UV++5Xa1VTrB6LvB\r\nrFyHNfweMEeXU9OE2KuEa+5KLrvk0S/9ycLjCNO0gwZkGAuZWGZYw4D9PS/fUk5xoZ+7sO2uWSXD\r\n2oWQ1OjSdtgH2IQg+/XX/4kacY8oiEEzYmKn4jMOREFvFuuQs77ToqghWlOLAgMBAAEwDQYJKoZI\r\nhvcNAQELBQADgYEAOQ3sTUGBLBn4B7ne1frQI0BOaOUdF0WOphD4zs2PM2MAx414H4WVas2nL9Rj\r\n+9nCQdzO9ZoHTQ+SCXdg8QmlFsDbXGsjN5L9mp5JVUWCJD/qWHx5RGvg7RVoirqm7K6bmejj5khk\r\nmLxFyUUFBPymorsIQ/X0l7Qn0aguvoxKGqw=\r\n-----END CERTIFICATE-----",
            "DEVICEMAXSPEED": 80.0,
            "DEVICETURNDEGREE": 45.0,
            "DEVICESENDDATAPERIOD": 3600.0,
            "DEVICESENDDATAINTERVALTIME": 10.0,
            "DEVICESENDDATAINTERVALDISTANCE": 500.0,
            "DEVICEMINACCELERATE": 13.0,
            "DEVICEAES": 0.0,
            "DEVICEODOMETERK": 800.0,
            "DEVICEDIGITALOUTPUT": 0.0,
            "BLUETOOTHADDRESSLIST": null,
            "COMPANYAGENTCODE": 2255.0,
            "COMPANYAGENTDESC": "هادي سروازاد (رادشيد)",
            "INSERTDATE": "2022-11-02T12:21:57",
            "BATCHNUMBER": 2.0,
            "COMPANYCODE": 10260544520.0
        }
    ]
    """


def call_get_logs(token: str, companycode: int, IMEI: str):
    body = {"token": token, "callback": "",
            "companycode": companycode, "Day": "01", "IMEI": IMEI}
    r = requests.post(
        "https://report.sipaad.ir/IGS.ITM.WebSiteServiceMiddle/WebServices/ReportsTest.asmx/GetLogs", json=body)
    return _strip_dnull(r.text)
    """لاگی برای این دستگاه در تاریخ مورد نظر یافت نشد"""


def call_get_events(token: str, IMEI: str):
    body = {"token": token,
            "callback": "", "IEMI": IMEI, "VIN": "", "PlaqueType": "", "PlaqueSN": "", "PlaqueID": "", "FromDate": "1402/07/25", "ToDate": "1402/07/25", "FromHour": "", "ToHour": "", "DeviceType": 1}
    r = requests.post(
        "https://report.sipaad.ir/IGS.ITM.WebSiteServiceMiddle/WebServices/ReportsTest.asmx/GetEvents", json=body)
    body = json.loads(_strip_dnull(r.text))
    _handle_err(body)
    return body
    """
    [
        {
            "latitude": 29.4421062469482,
            "longitude": 51.7873268127441,
            "EventTime": "2023-10-17T10:02:18",
            "EventType": "0x02",
            "EventTypeDesc": "نقطه معمولی",
            "GPSSpeed": 73,
            "GPSMaxSpeed": 73,
            "GPSTotalTraveledDistance": 225,
            "GPSStatus": "0x00",
            "Altitude": 782.0,
            "Bearing": 141.0,
            "NumberOfSatellite": 18,
            "PDOP": 10,
            "ECUSpeed": 0,
            "ECUMaxSpeed": 0,
            "ECUTotalTraveledDistance": 0,
            "IOStatus": "0xE0",
            "HighResolutionFuelConsumption": 0.0,
            "FuelLevel": 0.0,
            "EngineSpeed": 0,
            "EngineTotalHourOfOperation": 0,
            "HighResolutionVehicleDistance": 0,
            "TachographSpeed": 0,
            "TachographStatus": "0x00000000",
            "EngineTemperature": -40,
            "AirSupplyPressure": 0,
            "TelltaleState": "0x0000000000000000",
            "CruiseControl": "0x00000000",
            "VehicleWeight": 0,
            "GSensorValue": 14,
            "SupplyVoltage": 122,
            "AnalogInput1": "0x0000",
            "AnalogInput2": "0x0000",
            "EventTimeUnix": 1697524338,
            "EventSource": 1
        }
    ]
    """


def call_login(username: str, password: str, captcha_id: str, captcha_value: str) -> IGSLoginInfo:
    body = {"callback": '', "UserType": 'deviceCompany', "UserName": username, "Password": password,
            "CaptchaID": captcha_id, "CaptchaValue": captcha_value}
    r = requests.post(
        "https://report.sipaad.ir/IGS.ITM.WebSiteServiceMiddle/WebServices/Authentication.asmx/LoginToSystem", json=body)
    body = json.loads(_strip_dnull(r.text))
    _handle_err(body)
    return IGSLoginInfo(body)
