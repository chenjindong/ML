def pickle_demo():
    '''
    pickle 序列化对象，以二进制的形式存储在磁盘
    '''
    import pickle
    d = {'a': [1,2,3], 'b': [4,5,6]}
    with open('data.pkl', 'wb') as output:
        pickle.dump(d, output)
    with open('data.pkl', 'rb') as fin:
        d = pickle.load(input)
    print(type(d))
    print(d)

# pickle_demo()

def json_demo():
    '''
    json 用于存储dict，以字符串的形式存储在磁盘
    '''
    import json
    d = {'a': [1, 2, 3], 'b': [4, 5, 6]}
    with open('data.txt', 'w', encoding='utf-8') as fin:
        json.dump(d, fin)
    with open('data.txt', 'r', encoding='utf-8') as fin:
        res = json.load(fin)
    print(type(res))
    print(res)
# json_demo()