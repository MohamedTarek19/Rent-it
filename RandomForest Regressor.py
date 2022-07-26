import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler,OneHotEncoder , MinMaxScaler
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


Data = pd.read_csv("train.csv", encoding= 'unicode_escape')
    
Data["Date"] = pd.to_datetime(Data["Date"])
# Data["Date"] = Data["Date"].dt.strftime('%m-%d-%Y')
Data["Date2"] = Data["Date"].dt.strftime('%d/%m/%Y')
Data["Date"] = pd.to_datetime(Data["Date2"])
Data = Data.drop(["Date2"],axis = 1)
x = Data["Date"]
months = x.dt.month
days = x.dt.day
years = x.dt.year
Data = Data.drop(["Date"],axis=1)


data = pd.concat([years,months,days,Data],axis = 1)


labEncoder1 = LabelEncoder()
labEncoder1.fit(["Autumn","Winter","Spring","Summer"])
data["Seasons"] = labEncoder1.fit_transform(data["Seasons"])
labEncoder = LabelEncoder()
data["Holiday"] = labEncoder.fit_transform(data["Holiday"])
data["Functioning Day"] = labEncoder.fit_transform(data["Functioning Day"])

onehotencoder = OneHotEncoder()
# [autumn:0 , winter:3, spring:1 , summer:2]

X = data.drop(["Rented Bike Count"],axis = 1)
Y = data["Rented Bike Count"].values

standTrans = StandardScaler()

minMax = MinMaxScaler()
X[['Visibility (10m)','Temperature(Â°C)','Dew point temperature(Â°C)','Humidity(%)','Wind speed (m/s)']] = minMax.fit_transform(X[['Visibility (10m)','Temperature(Â°C)','Dew point temperature(Â°C)','Humidity(%)','Wind speed (m/s)']])

X = X.values

param_grid = { 
    'n_estimators': [100,120,140,160,180],
    #'max_depth' : [4,5,6,7,8],
    'max_features' :[4,5,6,7,8],
    'bootstrap'    : [True, False],
}
#rfg = RandomForestRegressor(random_state = 42)
#CV_rfc = GridSearchCV(estimator=rfg, param_grid=param_grid, cv= 5)
#CV_rfc.fit(X, Y)
#print(CV_rfc.best_params_)


model = RandomForestRegressor(n_estimators = 140,max_features = 4 ,bootstrap = True, random_state=0)
model.fit(X,Y)
###########################[test dataset]#############################
testData = pd.read_csv("test.csv", encoding= 'unicode_escape')

testData["Date"] = pd.to_datetime(testData["Date"])
# Data["Date"] = Data["Date"].dt.strftime('%m-%d-%Y')
testData["Date2"] = testData["Date"].dt.strftime('%d/%m/%Y')
testData["Date"] = pd.to_datetime(testData["Date2"])
testData = testData.drop(["Date2"],axis = 1)
x = testData["Date"]
months = x.dt.month
days = x.dt.day
years = x.dt.year
tData = testData.drop(["Date"],axis=1)

tData = pd.concat([years,months,days,tData],axis = 1)


ID = tData["ID"].values
tData = tData.drop(["ID"],axis = 1)

tData["Seasons"] = labEncoder1.fit_transform(tData["Seasons"])
tData["Holiday"] = labEncoder.fit_transform(tData["Holiday"])
tData["Functioning Day"] = labEncoder.fit_transform(tData["Functioning Day"])
test_X= tData


test_X[['Visibility (10m)','Temperature(Â°C)','Dew point temperature(Â°C)','Humidity(%)','Wind speed (m/s)']] = minMax.fit_transform(test_X[['Visibility (10m)','Temperature(Â°C)','Dew point temperature(Â°C)','Humidity(%)','Wind speed (m/s)']])
# Xnor = minMax.fit(test_X)
# test_X = Xnor.transform(test_X)
test_X= test_X.values

Y_pred = model.predict(test_X)
# Y_pred = abs(Y_pred)
# print(type(Y_pred[0]))
with open('pred.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    header = ['ID','Rented Bike Count']
    writer.writerow(header)
    x =[]
    # write a row to the csv file
    for i in range(0,len(Y_pred)):
        x.append([str(ID[i]),int(Y_pred[i])])
    writer.writerows(x)
# close the file
    f.close()

score = model.score(X,Y)
print(score)