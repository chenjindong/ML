import pandas as pd

time = []
for hour in [15, 16, 17]:
    for minute in range(6):
        time.append('2016-09-14-' + str(hour) + '-' + str(minute))
# df = pd.DataFrame({'time': time})
# df.to_csv('C:\\Users\\cjd\\Desktop\\airport\\testSet\\time.csv')

wifi_ap = pd.read_csv('C:\\Users\\cjd\\Desktop\\airport\\WIFI_AP_Passenger_Records_chusai_1stround.csv')
result01 = wifi_ap[(wifi_ap['timeStamp'] > '2016-09-12-15') & (wifi_ap['timeStamp'] < '2016-09-12-18')]
result01 = result01.sort_values(['WIFIAPTag', 'timeStamp'])
result02 = wifi_ap[(wifi_ap['timeStamp'] > '2016-09-13-15') & (wifi_ap['timeStamp'] < '2016-09-13-18')]
result02 = result02.sort_values(['WIFIAPTag', 'timeStamp'])

# result.to_csv('C:\\Users\\cjd\\Desktop\\airport\\wifi_ap_preprocess\\data01.csv')

passengerCount01 = result01.passengerCount.tolist()
passengerCount02 = result02.passengerCount.tolist()
WIFIAPTag = wifi_ap.WIFIAPTag.unique().tolist()

data = []
idx = 0
for col1 in WIFIAPTag:
    for col2 in time:
        temp = 0
        for i in range(10):
            temp = temp + passengerCount01[idx] + passengerCount02[idx]
            idx += 1
        data.append([temp/20.0, col1, col2])
df = pd.DataFrame(data, columns=['passengerCount', 'WIFIAPTag', 'slice10min'])
df.to_csv('C:\\Users\\cjd\\Desktop\\airport\\testSet\\airport_gz_passenger_predict02.csv')




