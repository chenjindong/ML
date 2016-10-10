import pandas as pd


data = pd.read_csv('C:\\Users\\cjd\\Desktop\\airport\\airport_gz_security_check_chusai_1stround.csv')

result = data.drop(['passenger_ID'], axis=1)

result = result.groupby(['flight_ID'])
# result.to_csv('C:\\Users\\cjd\\Desktop\\airport\\security_check\\result.csv')
print(result.count())