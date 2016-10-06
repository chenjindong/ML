import pandas as pd

security_check = pd.read_csv('C:\\Users\\cjd\\Desktop\\airport\\airport_gz_security_check_chusai_1stround.csv')
print(security_check.info())
# print(data.isnull().sum())
print(len(security_check['flight_ID'].unique()))
print('-------------------------------------------------------------------------')
departure = pd.read_csv('C:\\Users\\cjd\\Desktop\\airport\\airport_gz_departure_chusai_1stround.csv')
print(departure.info())
print(len(departure['flight_ID'].unique()))
print('-------------------------------------------------------------------------')
wifi_ap = pd.read_csv('C:\\Users\\cjd\\Desktop\\airport\\WIFI_AP_Passenger_Records_chusai_1stround.csv')
print(wifi_ap.info())