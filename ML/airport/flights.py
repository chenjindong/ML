import pandas as pd

data = pd.read_csv('C:\\Users\\cjd\\Desktop\\airport\\airport_gz_flights_chusai_1stround.csv')

result = data.dropna(how='any').copy() # 删除有空值的列
# result.to_csv('C:\\Users\\cjd\\Desktop\\airport\\flight\\flight.csv')
result = result.drop(['scheduled_flt_time'], axis=1)

l = result.flight_ID.count()
for i in result.index.tolist():
    time = result.loc[i, 'actual_flt_time']
    if len(time) != 18:
        temp = time[0:10]+'0'+time[10:]
    else:
        temp = time
    result.loc[i, 'actual_flt_time'] = temp
    if len(temp) != 18:
        result = result.drop(i)  # 删除某行
result = result.drop(691)

result = result[(result.actual_flt_time > '2016/9/14 07') & (result.actual_flt_time < '2016/9/14 10')]

gate = pd.read_csv('C:\\Users\\cjd\\Desktop\\airport\\airport_gz_gates.csv')

result = pd.merge(result, gate, left_on='BGATE_ID', right_on='BGATE_ID')

result.to_csv('C:\\Users\\cjd\\Desktop\\airport\\flight\\flight.csv')
