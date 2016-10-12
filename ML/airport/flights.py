import pandas as pd

data = pd.read_csv('C:\\Users\\cjd\\Desktop\\airport\\airport_gz_flights_chusai_1stround.csv')

result = data.dropna(how='any') # 删除有空值的列
# result.to_csv('C:\\Users\\cjd\\Desktop\\airport\\flight\\flight.csv')


actual_flt_time = result.actual_flt_time.tolist()
newcolumn = []
for time in actual_flt_time:
    if len(time) != 18:
        newcolumn.append(time[0:10]+'0'+time[10:])
        # print(time[0:10]+'0'+time[10:])
    elif len(time) == 18:
        newcolumn.append(time)
    else:
        print('hahah')
print(len(newcolumn))
result = result.drop(['actual_flt_time'], axis=1)
print(newcolumn)
df = pd.DataFrame({'actual_flt_time': newcolumn}, columns=['actual_flt_time'])
result = result.join(df)
print(result.info())
result.to_csv('C:\\Users\\cjd\\Desktop\\airport\\flight\\flight.csv')

