import pandas as pd

data = pd.read_csv('C:\\Users\\cjd\\Desktop\\airport\\airport_gz_flights_chusai_1stround.csv')
print(len(data.flight_ID.unique()))