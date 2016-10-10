import numpy as np
import pandas as pd
import re

# DataFrame is a two-dimensional table

data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}
df = pd.DataFrame(data,  columns=['state', 'year', 'pop', 'debt'])
# print(df.info())  # get the data type and non-null data number
# print(type(df.values))
# print(df)




