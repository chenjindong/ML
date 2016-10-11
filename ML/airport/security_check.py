import pandas as pd


data = pd.read_csv('C:\\Users\\cjd\\Desktop\\airport\\airport_gz_security_check_chusai_1stround.csv')

result = data.drop(['passenger_ID'], axis=1)

grouped_result = result.groupby(['flight_ID']).count()
flight = grouped_result[grouped_result.security_time > 20].index.tolist()
# print(flight)
result = result[result.flight_ID .isin(flight)]
grouped_result = result.groupby(['flight_ID'])
for name, group in grouped_result:
    if name == 'ZH9764':
        print(group)

# result.to_csv('C:\\Users\\cjd\\Desktop\\airport\\security_check\\result.csv')
# print(result.count())
