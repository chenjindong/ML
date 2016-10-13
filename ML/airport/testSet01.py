import pandas as pd

flight = pd.read_csv('C:\\Users\\cjd\\Desktop\\airport\\flight\\flight.csv')
predict = pd.read_csv('C:\\Users\\cjd\\Desktop\\airport\\testSet\\airport_gz_passenger_predict.csv')

for i in flight.index.tolist():
    ftime, fgate = flight.loc[i, 'actual_flt_time'], flight.loc[i, 'BGATE_AREA']
    ftime_num = (int(ftime[10])+8)*6+int(ftime[12])
    for j in predict.index.tolist():
        ptime, pgate = predict.loc[j, 'slice10min'], predict.loc[j, 'WIFIAPTag'][0:2]
        ptime_num = (int(ptime[11:13]))*6+int(ptime[14])
        if fgate == pgate:
            if (ftime_num == ptime_num) & (predict.loc[j, 'passengerCount'] > 0.5):
                predict.loc[j, 'passengerCount'] -= 0.5
            temp = ftime_num-ptime_num
            if (temp > 0) & (temp < 6):
                predict.loc[j, 'passengerCount'] += (6-temp)*0.1

predict.to_csv('C:\\Users\\cjd\\Desktop\\airport\\testSet\\airport_gz_passenger_predict01.csv')


