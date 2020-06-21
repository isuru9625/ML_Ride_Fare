import pandas as pd
import numpy as np
data = pd.read_csv('train.csv')
data['lat_diff'] = data['drop_lat'] - data['pick_lat']
data['lon_diff'] = data['drop_lon'] - data['pick_lon']
df2 = pd.DataFrame()
df2["lat_diff"]  = data["lat_diff"]
df2["lon_diff"]  = data["lon_diff"]
list1 =[]
for i in range(0,len(df2.lat_diff)):
    cal = df2.lat_diff[i]**2 + df2.lon_diff[i]**2 
    cal = cal**0.5
    list1.append(cal)
df2["distance"] = pd.Series(list1)
data["distance"] =df2["distance"]
data["pickup_time"] = pd.to_datetime(data.pickup_time)
data["drop_time"] = pd.to_datetime(data.drop_time)
data["pickup_year"] = data.pickup_time.dt.year
data["pickup_month"] = data.pickup_time.dt.month
data["pickup_date"] = data.pickup_time.dt.day
data["pickup_hour"] = data.pickup_time.dt.hour
data["pickup_minute"] = data.pickup_time.dt.minute
data["drop_year"] = data.drop_time.dt.year
data["drop_month"] = data.drop_time.dt.month
data["drop_date"] = data.drop_time.dt.day
data["drop_hour"] = data.drop_time.dt.hour
data["drop_minute"] = data.drop_time.dt.minute
from datetime import date
from datetime import datetime
d0 = date(data.pickup_year[0],data.pickup_month[0],data.pickup_date[0])
d1 = date(data.drop_year[0],data.drop_month[0],data.drop_date[0])
date_format = "%m/%d/%Y"
df = pd.DataFrame()
df["number_of_nights"] = data['drop_time'] - data['pickup_time']
df['actual_seconds'] = df.number_of_nights.dt.total_seconds()
data["seconds"] = df["actual_seconds"]
df['nights'] = pd.Series(delta.days for delta in (df['number_of_nights']))
df['seconds'] = pd.Series(delta.seconds for delta in (df['number_of_nights']))
data["number_of_nights"] = df["nights"]
list1 = list(data.number_of_nights)
data["pickup_day"] = data.pickup_time.apply(lambda x: x.weekday())
data["seconds"] = df["seconds"]
duration_error = data[data.seconds+60<data.duration]
duration_error_list = duration_error[duration_error.number_of_nights==0].index
for i in range(0,len(duration_error_list)):
    data.duration[duration_error_list[i]] = data.seconds[duration_error_list[i]]

data.to_csv("features/train_features.csv")
