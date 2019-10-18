
import requests
import json
import cjdpy
import time
import os


def get_osAccessToken():
    url = "https://api.xiaoyastar.com/serv/account/os-access-token?" \
          "productId=S_PROD1_593&osServerSecret=f94016e41c49474b805e0528463dd2bf"
    headers = {
        "Postman-Token": "d1c79404-da26-4ee9-ad1e-64ef0a9a4c44",
        "cache-control": "no-cache",
    }
    resp = requests.get(url, headers=headers) #
    resp_json = json.loads(resp.text)
    return resp_json["osAccessToken"]

def nlu_from_xmly(osAccessToken, query):

    url = 'https://api.xiaoyastar.com/serv/text/query'
    headers = {
        "Postman-Token": "b0b6ee44-e70d-4b20-90c4-5db01dcb5334",
        "cache-control": "no-cache",
    }
    data = {"osAccessToken": osAccessToken,
            "params": '{"osOpenId": "4dc36d8e546c41aaa717552bfd43ac09", "deviceType":2, "productId": "S_PROD1_593","sn": "Ejfi339Jsg", "lat":"1.1", "lng":"11.1","appVersion":"1.0","sysVersion":"1.0", "speakerVersion":"1.0", "romVersion":"1.0", "dt":1571122149764}',
            "text": query}
    resp = requests.get(url, headers=headers, params=data)
    return json.loads(resp.text)

