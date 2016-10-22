import pandas as pd

time = []
for hour in [15, 16, 17]:
    for minute in range(6):
        time.append('2016-09-25-' + str(hour) + '-' + str(minute))
df = pd.DataFrame({'time': time})
df.to_csv('C:\\Users\\cjd\\Desktop\\airport\\testSet\\time.csv')

wifi_ap = pd.read_csv('C:\\Users\\cjd\\Desktop\\airport\\WIFI_AP_Passenger_Records_chusai_2ndround.csv')
# print(wifi_ap.info())


result01 = wifi_ap[(wifi_ap['timeStamp'] > '2016-09-12-15') & (wifi_ap['timeStamp'] < '2016-09-12-18')]
result01 = result01.sort_values(['WIFIAPTag', 'timeStamp'])
result02 = wifi_ap[(wifi_ap['timeStamp'] > '2016-09-13-15') & (wifi_ap['timeStamp'] < '2016-09-13-18')]
result02 = result02.sort_values(['WIFIAPTag', 'timeStamp'])
result03 = wifi_ap[(wifi_ap['timeStamp'] > '2016-09-11-15') & (wifi_ap['timeStamp'] < '2016-09-11-18')]
result03 = result03.sort_values(['WIFIAPTag', 'timeStamp'])  # 按多列进行排序

result01.to_csv('C:\\Users\\cjd\\Desktop\\airport\\wifi_ap_preprocess\\result01.csv')
result02.to_csv('C:\\Users\\cjd\\Desktop\\airport\\wifi_ap_preprocess\\result02.csv')
result03.to_csv('C:\\Users\\cjd\\Desktop\\airport\\wifi_ap_preprocess\\result03.csv')

passengerCount01 = result01.passengerCount.tolist()
passengerCount02 = result02.passengerCount.tolist()
passengerCount03 = result03.passengerCount.tolist()
print(len(passengerCount01), len(passengerCount02), len(passengerCount03))
WIFIAPTag = wifi_ap.WIFIAPTag.unique().tolist()
print(len(WIFIAPTag))

data = []
idx = 0
for col1 in WIFIAPTag:
    for col2 in time:
        temp = 0
        for i in range(10):
            temp = temp + passengerCount01[idx] + passengerCount02[idx] + passengerCount03[idx]
            idx += 1
        data.append([temp/30.0, col1, col2])
df = pd.DataFrame(data, columns=['passengerCount', 'WIFIAPTag', 'slice10min'])
# df.to_csv('C:\\Users\\cjd\\Desktop\\airport\\testSet\\airport_gz_passenger_predict.csv')









