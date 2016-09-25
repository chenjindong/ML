import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

train = pd.read_csv("C:\\Users\\cjd\\Desktop\\titanic\\train.csv")

# The relation between Social status and survived
print(train["Pclass"].isnull().value_counts())
survivors = train.groupby("Pclass")["Survived"].agg(sum)
allPassenger = train.groupby('Pclass').count()['Survived']
print(survivors)
print(allPassenger)
rate = survivors/allPassenger
fig = plt.figure()
ax = fig.add_subplot(111)
rect = ax.bar(rate.index.values.tolist(), rate, color='blue', width=0.3)
ax.set_ylabel('No. of survivors')
ax.set_title('Total number of survivors based on class')
plt.show()

# print(train['Sex'].isnull().value_counts())
# survivors = train.groupby('Sex')['Survived'].agg(sum)
# print(survivors)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_ylabel('No. of survivors')
# ax.set_xlabel('female                                                                 male')
# rect = ax.bar([1, 2], survivors, color='blue', width=0.3)
# ax.set_title('Total number of survivors based on class')
# plt.show()
