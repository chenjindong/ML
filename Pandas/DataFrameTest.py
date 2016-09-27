import numpy as np
import pandas as pd

# DataFrame is a two-dimensional table

data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}
df = pd.DataFrame(data,  columns=['state', 'year', 'pop', 'debt'])  # you can specify the order of column by parameter columns, and the index by parameter index
# print(df)
# print(df['state'])  # print the column of state
# print(df.state)
# print(df.ix[2])  # print the row of 2

# df['debt'] = np.arange(5)  # assign value for column debt
# print(df)

# df['eastern'] = df['state'] == 'Ohio'  # add a new column
# print(df)
# del df['eastern']  # delete a column
# print(df)

# print('state' in df.columns)
# print(2 in df.index)
# print(5 in df.index)

# dummy non-numerical value
obj1 = pd.DataFrame({'Month': ['monday', 'tuesday', 'wednesday']})
# print(obj1.describe)
month = pd.get_dummies(obj1['Month'])
# print(month.columns.values)
# print(month.values)
# obj1[month.columns.values] = month.values
for i in [0, 1, 2]:
        obj1[month.columns.values[i]] = month.values[i]

print(obj1.drop(['Month'], axis=1))
print(obj1)

