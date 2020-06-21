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
data.to_csv("Missing_Values_Handling/final_test.csv")
