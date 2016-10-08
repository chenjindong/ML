import pandas as pd

time = []
for hour in [15, 16, 17]:
    for minute in range(6):
        time.append('2016-09-14-' + str(hour) + '-' + str(minute))
# df = pd.DataFrame({'time': time})
# df.to_csv('C:\\Users\\cjd\\Desktop\\airport\\testSet\\time.csv')

wifi_ap = pd.read_csv('C:\\Users\\cjd\\Desktop\\airport\\WIFI_AP_Passenger_Records_chusai_1stround.csv')
WIFIAPTag = wifi_ap.WIFIAPTag.unique().tolist()
data = []
for col1 in WIFIAPTag:
    for col2 in time:
        data.append([15, col1, col2])
df = pd.DataFrame(data, columns=['passengerCount', 'WIFIAPTag', 'slice10min'])
df.to_csv('C:\\Users\\cjd\\Desktop\\airport\\testSet\\airport_gz_passenger_predict.csv')




