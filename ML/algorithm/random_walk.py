import random

def random_walk(n):
    """Return coordinates after 'n' random walk"""
    x = 0
    for i in range(n):
        step = random.choice(['P', 'N'])  # toss a coin=>positive or nagative
        if step == 'P':
            x += 1
        else:
            x -=1
    return x

for i in range(10):  # test 10 times
    x = random_walk(100)  # random wolk 100 step
    print(x)

