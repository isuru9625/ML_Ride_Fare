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
