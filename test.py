# import pandas as pd
# df = pd.read_excel("C:\\Users\\cjd\\Desktop\\test.xls")
# print(df.describe())


# 函数传参问题
# def add(a):
#     a.append(1)
#
# a = []
# add(a)
# print(a)

# list不需要global声明，普通变量需要global声明.
def show():
    #global a
   # a = 12
    print(a)
    b.append(10)
    print(b)

if __name__ == '__main__':
    a = 10
    b = [10]
    show()
    # print(a)
    print(b)















