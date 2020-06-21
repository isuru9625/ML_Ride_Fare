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
