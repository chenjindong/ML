import os
import requests
import re

def RM(patt, sr):
    mat = re.search(patt, sr, re.DOTALL | re.MULTILINE)
    return mat.group(1) if mat else ''


def get_page(url, cookie='', proxy=''):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/42.0.2311.90 Safari/537.36'}
        if cookie != '': headers['cookie'] = cookie
        if proxy != '':
            proxies = {'http': proxy, 'https': proxy}
            resp = requests.get(url, headers=headers, proxies=proxies, timeout=5.0)
        else:
            resp = requests.get(url, headers=headers, timeout=5.0)
        content = resp.content
        headc = content[:min([3000, len(content)])].decode(errors='ignore')
        charset = RM('charset="?([-a-zA-Z0-9]+)', headc)
        if charset == '': charset = 'utf-8'
        content = content.decode(charset, errors='replace')
    except Exception as e:
        print(e)
        content = ''
    return content

def get_files(path):
    '''递归读取path目录下的所有文件名'''
    pairs = []
    for tup in os.walk(path):
        for item in tup[2]:
            path = tup[0] + '\\' + item
            pairs.append(path)
    print('get files done.')
    return pairs

def show_status(num, interval):
    if num % interval == 0:
        print('processing line %d' % num)

def is_Chinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False

def load_set(fn, ):
    return set(load_list(fn))

def load_csv_row(fn, num=0, separator='\t'):
    res = []
    with open(fn, 'r', encoding='utf-8') as fin:
        for line in fin:
            res.append(line.strip().split(separator)[num])
    return res

def load_list(fn):
    res = []
    with open(fn, 'r', encoding='utf-8') as fin:
        for line in fin:
            res.append(line.strip())
    return res

def load_csv(fn, separator='\t'):
    res = []
    with open(fn, 'r', encoding='utf-8') as fin:
        for line in fin:
            res.append(line.strip().split(separator))
    return res

def load_dict(fn, func=str):
    res = {}
    with open(fn, 'r', encoding='utf-8') as fin:
        for line in fin:
            items = line.strip().split('\t')
            if len(items) == 2:
                res[items[0]] = func(items[1])
    return res


def save_lst(lst, fn):
    with open(fn, 'w', encoding='utf-8') as fout:
        for item in lst:
            fout.write(str(item) + '\n')
            fout.flush()

def save_dict(d, fn):
    save_csv(d.items(), fn)

def save_csv(lst, fn):
    with open(fn, 'w', encoding='utf-8') as fout:
        for line in lst:
            fout.write('\t'.join([str(x) for x in line]) + '\n')
            fout.flush()



