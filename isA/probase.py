#coding:utf-8
import json
import re
from urllib.request import Request, urlopen
import urllib.parse


def google_tran(source):
    url = r'https://translate.google.cn/'
    data = {'hl': 'zh-CN', 'ie': 'UTF-8', 'text': source, 'langpair': "'en'|'zh-CN'"}
    url_values = urllib.parse.urlencode(data, encoding='utf-8')
    full_url = url + '?' + url_values
    req = Request(full_url, headers={'User-Agent': 'Mozilla/5.0'})
    html = urlopen(req).read().decode('utf-8')
    p = re.compile("(?<='#fff'\">).*?<")    #(?<=string) 匹配字符串string    '#fff'">苹果<
    m = p.search(html)
    target = m.group(0).strip('<')
    return target


def get_concepts(entity):
    data = {}
    data['kw'] = entity
    data['start'] = 0
    url_values = urllib.parse.urlencode(data, encoding='utf8')
    url = 'http://knowledgeworks.cn:20314/probaseplus/pbapi/getconcepts'
    full_url = url + '?' + url_values
    with urllib.request.urlopen(full_url) as response:
        res = response.read().decode('utf-8')
        resdict = json.loads(res)
        return resdict['concept']

def main():
    concepts = get_concepts('mongodb')
    for item in concepts:
        target = google_tran(item[0])
        print(item[0], target)


if __name__ == '__main__':
    main()
