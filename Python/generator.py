
'''
生成器
'''
def generate_train_data(n):
    cnt = 1
    while(cnt <= n):
        yield [cnt, cnt+1]
        cnt += 2

#----------------method1------------------
# g是生成器对象
g = generate_train_data(10)
# 取生成器的第1个对象，执行到yield语句，等待下一次执行
print(g.__next__())
# 取生成器的第二个对象
print(g.__next__())

#----------------method2------------------
for item in generate_train_data(10):
    print(item)
