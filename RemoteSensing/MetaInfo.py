import os
from PIL import Image
import pandas as pd


def getTFW():
    data=[]
    filepath = "D:\CJD\Data\SAR\ArcticDecompression"
    pathDir = os.listdir(filepath)
    for name in pathDir:
        path = filepath + '\\' + name
        if name[-1] == 'w':
            f = open(path)
            lines = f.readlines()
            row = [name.split('.')[0]]
            for line in lines:
                cnt = line.split('\n')
                row.append(float(cnt[0]))
            data.append(row)
            f.close()
        if name[-1] =='f':
            img = Image.open(path)
            data[-1].append(img.size[0])
            data[-1].append(img.size[1])
    print(len(data))
    print(data)
    res = pd.DataFrame(data, columns=['name', 'A', 'D', 'B', 'E', 'C', 'F', 'width', 'height'])
    res.to_csv("C:\\Users\\cjd\\Desktop\\northtwf.csv")

def getImageInfo():
    indata = pd.read_csv("C:\\Users\\cjd\\Desktop\\northtwf.csv")
    data = indata.values
    result = []
    # name satellite time top left bottom right south(tag) jpg_url tif_url
    for record in data:
        row = []
        row.append(record[0])  # name
        row.append('sentinel-1')  # satellite
        time = record[0].split('_')[4]
        row.append(time[0:4]+'-'+time[4:6]+'-'+time[6:8]+' '+time[9:11]+':'+time[11:13]+':'+time[13:15])  # time
        row.append(record[6])  # top
        row.append(record[5])  # left
        bottom = record[2]*record[7] + record[4]*record[8] + record[6]
        right = record[1]*record[7] + record[3]*record[8] + record[5]
        row.append(bottom)  # bottom
        row.append(right)  # right
        row.append(0)  # south tag
        row.append('image/png/'+record[0]+'.jpg')  # jpg_url
        row.append('image/tif/'+record[0]+'.tif.tar.gz')  # tif_url 还得修改，改成tar.gz
        result.append(row)
    res = pd.DataFrame(result, columns=['name', 'satellite', 'time', 'otop', 'oleft', 'obottom',
                                        'oright', 'south', 'png_url', 'tif_url'])
    res.to_csv("C:\\Users\\cjd\\Desktop\\northproject.csv")

# getTFW()
getImageInfo()
