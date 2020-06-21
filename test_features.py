import pandas as pd
import numpy as np
from datetime import date
from datetime import datetime
d_test = pd.read_csv("test.csv")
d_test["pickup_time"] = pd.to_datetime(d_test.pickup_time)
d_test["drop_time"] = pd.to_datetime(d_test.drop_time)
d_test["pickup_year"] = d_test.pickup_time.dt.year
d_test["pickup_month"] = d_test.pickup_time.dt.month
d_test["pickup_date"] = d_test.pickup_time.dt.day
d_test["pickup_hour"] = d_test.pickup_time.dt.hour
d_test["pickup_minute"] = d_test.pickup_time.dt.minute
d_test["drop_year"] = d_test.drop_time.dt.year
d_test["drop_month"] = d_test.drop_time.dt.month
d_test["drop_date"] = d_test.drop_time.dt.day
d_test["drop_hour"] = d_test.drop_time.dt.hour
d_test["drop_minute"] = d_test.drop_time.dt.minute

date_format = "%m/%d/%Y"
df1 = pd.DataFrame()
df1["number_of_nights"] = d_test['drop_time'] - d_test['pickup_time']
df1['nights'] = pd.Series(delta.days for delta in (df1['number_of_nights']))
d_test["number_of_nights"] = df1["nights"]
df1['seconds'] = pd.Series(delta.seconds for delta in (df1['number_of_nights']))
df1['actual_seconds'] = df1.number_of_nights.dt.total_seconds()
d_test["seconds"] = df1["actual_seconds"]
d_test["pickup_day"] = d_test.pickup_time.apply(lambda x: x.weekday())
d_test['lat_diff'] = d_test['drop_lat'] - d_test['pick_lat']
d_test['lon_diff'] = d_test['drop_lon'] - d_test['pick_lon']
df3 = pd.DataFrame()
df3["lat_diff"]  = d_test["lat_diff"]
df3["lon_diff"]  = d_test["lon_diff"]
list2 =[]
for i in range(0,len(df3.lat_diff)):
    cal = df3.lat_diff[i]**2 + df3.lon_diff[i]**2 
    cal = cal**0.5
    list2.append(cal)
df3["distance"] = pd.Series(list2)
d_test["distance"] =df3["distance"]
d_test.duration[5491] = 780

d_test.to_csv("features/test_features.csv")
