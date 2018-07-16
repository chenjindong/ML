import os
from collections import Counter

def get_files(path):
    '''递归读取path目录下的所有文件名'''
    pairs = []
    for tup in os.walk(path):
        for item in tup[2]:
            path = tup[0] + '\\' + item
            pairs.append(path)
    print('get files done.')
    return pairs

def load_set(fn):
    return set(load_list(fn))

def load_csv_row(fn, num=0, separator='\t'):
    res = []
    with open(fn, 'r', encoding='utf-8', errors='ignore') as fin:
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
    with open(fn, 'r', encoding='utf-8', errors='ignore') as fin:
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


def sort_list_by_freq(lst, rev=True):
    '''rev：升序or降序，default as 升序'''
    counter = Counter(lst)
    return sorted(counter.items(), key=lambda x:x[1], reverse=rev)