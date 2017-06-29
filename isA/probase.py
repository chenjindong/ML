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
    # print(html) #-------------
    p = re.compile("(?<='#fff'\">).*?<")    #(?<=string) 匹配字符串string    '#fff'">苹果<
    m = p.search(html)
    if m == None:
        return ''
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

def get_entity_list():
    with open('it_skill.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        return [line.strip() for line in lines]

def main(entity_list):
    outfile = open('output1.txt', 'a', encoding='utf-8')
    for i in range(len(entity_list)):
        if i % 10 == 0:
            print('--------------%d---------------------' %(i))
        concepts = get_concepts(entity_list[i])
        if len(concepts) > 0:
            targets = []
            for item in concepts:
                temp = item[0]
                for ch in item[0]:
                    if ch >= 'a' and ch <= 'z':
                        temp = google_tran(item[0])
                        break
                targets.append(temp)
            for target in targets:
                outfile.write(entity_list[i] + '\t' + target + '\n')
                outfile.flush()
    outfile.close()

if __name__ == '__main__':
    entity_list = get_entity_list()
    main(entity_list)




