import pandas as pd


security_check = pd.read_csv('C:\\Users\\cjd\\Desktop\\airport\\airport_gz_security_check_chusai_1stround.csv')
print(security_check.info())
print('flight number-----------------', len(security_check['flight_ID'].unique()))

result = security_check[(security_check.security_time > '2016/9/13') & (security_check.security_time < '2016/9/14')]
print(len(result.flight_ID.unique()))
result.to_csv('C:\\Users\\cjd\\Desktop\\airport\\2016-9-13.csv')