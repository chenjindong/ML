import pandas as pd

data = pd.read_csv('C:\\Users\\cjd\\Desktop\\Sentiment_Analysis_Dataset_raw.csv', error_bad_lines=False)
print(data.count())
#data = data.drop(['SentimentSource', '﻿ItemID'], axis=1)

#data.to_csv('C:\\Users\\cjd\\Desktop\\tweets.csv')
