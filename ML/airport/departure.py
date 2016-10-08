import pandas as pd

departure = pd.read_csv('C:\\Users\\cjd\\Desktop\\airport\\airport_gz_departure_chusai_1stround.csv')
print(departure.info())
print('flight number----------- ', len(departure['flight_ID'].unique()))

# result = departure[(departure.flight_ID == 'HU7002') & (departure.flight_time < '2016/9/12')]
# result.to_csv('C:\\Users\\cjd\\Desktop\\airport\\departure_HU7002.csv')

print(departure[departure.checkin_time < '2016/9/11'])