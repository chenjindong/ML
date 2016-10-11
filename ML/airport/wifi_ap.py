import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


wifi_ap = pd.read_csv('C:\\Users\\cjd\\Desktop\\airport\\WIFI_AP_Passenger_Records_chusai_1stround.csv')
# tags = wifi_ap[(wifi_ap.WIFIAPTag < 'E3') & (wifi_ap.WIFIAPTag > 'E2')].WIFIAPTag.unique()
# print(tags)


# print(wifi_ap.info())
# print(wifi_ap.tail())

result = wifi_ap[(wifi_ap.WIFIAPTag > 'W1') & (wifi_ap.WIFIAPTag < 'W3')]
result.to_csv('C:\\Users\\cjd\\Desktop\\airport\\wifi_ap_preprocess\\data01.csv')


# x = np.arange(0, 4000, 1)
# left = 0
# right = 4000
#
# y = wifi_ap[wifi_ap.WIFIAPTag == 'E1-1A-1<E1-1-01>'].passengerCount[left:right]
# plt.plot(x, y, color='red')
#
# y = wifi_ap[wifi_ap.WIFIAPTag == 'E2-3B<E2-3-20>'].passengerCount[left:right]
# plt.plot(x, y, color='green')
#
# y = wifi_ap[wifi_ap.WIFIAPTag == 'E2-3B<E2-3-21>'].passengerCount[left:right]
# plt.plot(x, y, color='blue')
# plt.grid()
# plt.show()

