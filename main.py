import os
if not os.path.exists('features'):
    os.makedirs('features')

if not os.path.exists('Missing_Values_Handling'):
    os.makedirs('Missing_Values_Handling')

if not os.path.exists('Outliers_Handling'):
    os.makedirs('Outliers_Handling')
	
if not os.path.exists('Submissions'):
    os.makedirs('Submissions')

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

import pandas as pd
import numpy as np
data = pd.read_csv("features/train_features.csv")
d_test = pd.read_csv("features/test_features.csv")
del data['Unnamed: 0']
data = data.drop([30,11049, 12687, 12898, 13544, 14043])
#How to find those indexes to delete, those details are in outliers.ipynb
data = data.drop([11952])
data.additional_fare[10084] =10.5
data = data.drop([920])
data.to_csv("Outliers_Handling/train_features.csv")
d_test.to_csv("Outliers_Handling/test_features.csv")


import pandas as pd
import numpy as np
data = pd.read_csv("Outliers_Handling/train_features.csv")
df2 = pd.DataFrame()
df2["additional_fare"] = data["additional_fare"]
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy='most_frequent')
df_filled = imputer.fit_transform(df2)
df1 = pd.DataFrame(df_filled)
df1 = df1.rename(columns={0:"additional_fare"})
data["additional_fare"] = df1["additional_fare"]
from sklearn.impute import KNNImputer
df = pd.DataFrame(data.duration)
df["pick_lat"] = data.pick_lat
df["pick_lon"] = data.pick_lon
df["drop_lat"] = data.drop_lat
df["drop_lon"] = data.drop_lon
df["pickup_hour"] = data.pickup_hour
df["number_of_nights"] = data.number_of_nights
df["pickup_day"] = data.pickup_day
df["distance"] = data.distance
df["meter_waiting"] = data.meter_waiting
imputer = KNNImputer(n_neighbors=9)
df_filled = imputer.fit_transform(df)
df1 = pd.DataFrame(df_filled)
df1 = df1.rename(columns={9:"meter_waiting"})
data["meter_waiting"] = df1["meter_waiting"]

df = pd.DataFrame(data.duration)
df["meter_waiting"] = data.meter_waiting
df["pickup_hour"] = data.pickup_hour
df["number_of_nights"] = data.number_of_nights
df["pickup_day"] = data.pickup_day
df["meter_waiting_fare"] = data.meter_waiting_fare
imputer = KNNImputer(n_neighbors=8)
df_filled = imputer.fit_transform(df)
df1 = pd.DataFrame(df_filled)
df1 = df1.rename(columns={5:"meter_waiting_fare"})
data["meter_waiting_fare"] = df1["meter_waiting_fare"]

df = pd.DataFrame(data.additional_fare)
df["pickup_hour"] = data.pickup_hour
df["number_of_nights"] = data.number_of_nights
df["pickup_day"] = data.pickup_day
df["additional_fare"] = data.additional_fare
df["meter_waiting_till_pickup"] = data.meter_waiting_till_pickup
imputer = KNNImputer(n_neighbors=4)
df_filled = imputer.fit_transform(df)
df1 = pd.DataFrame(df_filled)
df1 = df1.rename(columns={4:"meter_waiting_till_pickup"})
data["meter_waiting_till_pickup"] = df1["meter_waiting_till_pickup"]

del data['Unnamed: 0']

df = pd.DataFrame(data.additional_fare)
df["duration"] = data.duration
df["meter_waiting"] = data.meter_waiting
df["meter_waiting_fare"] = data.meter_waiting_fare
df["fare"] = data.fare
imputer = KNNImputer(n_neighbors=6)
df_filled = imputer.fit_transform(df)
df1 = pd.DataFrame(df_filled)
df1 = df1.rename(columns={4:"fare"})
data["fare"] = df1["fare"]
data.to_csv("Missing_Values_Handling/final_train.csv")

d_test = pd.read_csv("Outliers_Handling/test_features.csv")
#Since no missing values are there
d_test.to_csv("Missing_Values_Handling/final_test.csv")

import pandas as pd
import numpy as np
data= pd.read_csv("Missing_Values_Handling/final_train.csv")
del data['Unnamed: 0']
features = ['additional_fare', 'duration', 'meter_waiting',
       'meter_waiting_fare', 'meter_waiting_till_pickup','number_of_nights', 'pickup_day','fare','distance']
X = data[features]
Y = data.label
from sklearn.model_selection import train_test_split
xTrain, xValid, yTrain, yValid = train_test_split(X, Y,train_size = 0.999, random_state=4)


from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn. ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
#model = MLPRegressor(hidden_layer_sizes=(10,5,),activation='logistic',max_iter=200)
model = RandomForestRegressor(n_estimators =12,max_features =7,max_depth =25, min_samples_split=3,min_samples_leaf=2, random_state=0)
model.fit(xTrain,yTrain)

d_test= pd.read_csv("Missing_Values_Handling/final_test.csv")

del d_test['Unnamed: 0']

features = ['additional_fare', 'duration', 'meter_waiting',
       'meter_waiting_fare', 'meter_waiting_till_pickup','number_of_nights', 'pickup_day','fare','distance']
X = d_test[features]
predictions = model.predict(X)

true_predictions = []
ones =0
zeros = 0
for i in range(0,len(predictions)):
    if(predictions[i]>0.7):
        true_predictions.append(1)
        ones=ones+1
    else:
        true_predictions.append(0)
        zeros =zeros+1

import csv
with open('Submissions/submission.csv', 'w+',newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['tripid','prediction'])
    for i in range(0,len(true_predictions)):
        
        writer.writerow([d_test.tripid[i],true_predictions[i]])


