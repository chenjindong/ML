import pandas as pd
import numpy as np

# Series is similar to list in python, or key-value pairs
# key is the index(索引)

# obj = pd.Series([1, 2, 3, 4])
# print(obj)
# print(obj.index)
# print(obj.values)

# obj1 = pd.Series([4, 7, -5, 1], index=['a', 'b', 'c', 'd'])
# print(obj1)
# print(obj1.index)
# print(obj1.values)
# print(obj1['b'])
# print(obj1[obj1 > 0])
# print(obj1*2)
# print(np.exp(obj1))
# print('a' in obj1)  # a is index

data = {'cjd': 23, 'slm': 22, 'bob': 18}
obj2 = pd.Series(data)
# print(obj2)
# print(obj2.isnull())
obj2.index = ['bob', 'steve', 'jeff']
print(obj2)







