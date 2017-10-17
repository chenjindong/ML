import matplotlib.pyplot as plt
from pylab import mpl
import numpy as np
from hyperparams import HyperParams as hp


def benchmark_precision():
    # 使中文能够正常显示
    # mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    # mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

    ticks_label_font = {'size': 12}
    axes_label_font = {'size': 12, 'weight': 'bold'}

    #  chinese x tick
    categories_file = open(hp.category_en_path, 'r', encoding='utf-8')
    categories, precisions = [], []
    fig, ax = plt.subplots()
    for line in categories_file.readlines():
        tuple = line.strip().split(':')
        categories.append(tuple[2])
        precisions.append(float(tuple[3]))


    x = [i*2 for i in range(40)]
    ax.bar(x, precisions, width=0.7)
    # 设置x轴(刻度、刻度标签)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right', fontdict=ticks_label_font)
    # 设置y轴(名称、刻度、刻度标签)
    ax.set_ylabel('Precision', fontdict=axes_label_font)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_yticklabels(['0%', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'], fontdict=ticks_label_font)
    # ax.yaxis.grid(grid_type='-')
    ax.grid(axis='y', linestyle='-')
    plt.show()
# benchmark_precision()
# assert False

def iteration_pairs():

    ticks_label_font = {'size': 14}
    axes_label_font = {'size': 12, 'weight': 'bold'}

    y = [128215, 185248, 257208, 311106, 327229, 327370, 327370, 327370, 327370, 327370, 327370]
    x = [i for i in range(1, 12, 1)]
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.scatter(x, y)
    # 设置x轴
    ax.set_xlabel('Iteration #', fontdict=axes_label_font)
    ax.set_xticks(x)
    # 设置y轴
    ax.set_ylabel('# of isA pairs', fontdict=axes_label_font)
    ax.set_yticks(np.arange(100000, 400000, 50000))
    ax.grid(axis='y', linestyle='-')
    plt.show()
# iteration_pairs()
# assert False

def iteration_precisions():

    ticks_label_font = {'size': 12}
    axes_label_font = {'size': 12, 'weight': 'bold'}

    chinese_isa_y = [0.940, 0.949, 0.954, 0.957, 0.957, 0.956, 0.956, 0.956, 0.956, 0.956, 0.956]
    probase_y = [0.973, 0.970, 0.956, 0.943, 0.937, 0.933, 0.932, 0.931, 0.930, 0.929, 0.928]
    english_isa_y = [0.965, 0.957, 0.950, 0.945, 0.940, 0.940, 0.940, 0.940, 0.940, 0.940, 0.940]
    x = [i for i in range(1, 12, 1)]
    fig, ax = plt.subplots()
    ax.plot(x, chinese_isa_y, label="Ours(Chinese)")
    ax.scatter(x, chinese_isa_y)
    ax.plot(x, probase_y, label="Probase")
    ax.scatter(x, probase_y)
    # ax.plot(x, english_isa_y, label="Ours")
    # ax.scatter(x, english_isa_y)

    ax.legend()
    # 设置x轴
    ax.set_xlabel('Iteration #', fontdict=axes_label_font)
    ax.set_xticks(x)
    # 设置y轴
    ax.set_ylabel('Precision', fontdict=axes_label_font)
    ax.set_yticks(np.arange(0.9, 1.01, 0.01))
    ax.set_yticklabels(['90%', '91%', '92%', '93%', '94%', '95%', '96%', '97%', '98%', '99%', '100%'], fontdict=ticks_label_font)
    ax.grid(axis='y', linestyle='-')
    plt.show()
iteration_precisions()





