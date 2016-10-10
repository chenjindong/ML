import pandas as pd

data = pd.read_csv('C:\\Users\\cjd\\Desktop\\airport\\airport_gz_departure_chusai_1stround.csv')

result = data.drop(['checkin_time'], axis=1)  # 删除列 取票时间

result = result.groupby(['flight_ID', 'flight_time'])
result = result.count()
# print(result.info())
result.rename(columns={'passenger_ID2': 'passenger_count'}, inplace=True)

result = result[result.passenger_count > 20]  # 去除无用数据[航班乘客数量小于20,则删除]

result.to_csv('C:\\Users\\cjd\\Desktop\\airport\\departure\\airport_gz_departure.csv')
