from multiprocessing import Pool, Manager
import time
import cjdpy
import jieba

'''
python中多线程just like a shit
python中的多进程相当于现实意义的多线程
run time:
    single thread: 53.56s
    5 multi-thread: 7.09s
    5 multi-thread (share memory=>Manage): 5.04s
'''
def single_thread():
    start = time.clock()
    for i in range(100000):
        words = list(jieba.cut(lines[i]))
    print(time.clock()-start)


def fun0(sentence):
    words = list(jieba.cut(sentence))

def simple_multi_thread():
    start=time.clock()
    pool = Pool(processes=5)
    for i in range(100000):
        pool.apply_async(fun0, args=(lines[i]))
    print(time.clock()-start)


def fun1(i):
    # share memory
    words = list(jieba.cut(manager_line[i]))

def share_memory_multi_thread():
    global manager_line
    manager_line = Manager().list()  # 让多个线程共享
    manager_line = lines
    start = time.clock()
    pool = Pool(processes=5)
    for i in range(100000):
        pool.apply_async(fun1, args=(i))
    print(time.clock()-start)

if __name__ == '__main__':
    path = r'\\10.141.208.22\data\Chinese_isA\corpus\wikicorpus.txt'
    lines = cjdpy.load_list(path)
    load_jieba = list(jieba.cut(lines[0]))

    single_thread()
    simple_multi_thread()
    share_memory_multi_thread()
