# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 14:05:21 2021

@author: DHANSHREE
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import pickle

data=pd.read_csv('DiamondPricesData.csv')
data=data.drop(['depth','table','x','y','z'],axis=1)
data['price']=data.price.astype(float)
data.dtypes
from sklearn.preprocessing import LabelEncoder
l1=LabelEncoder()
label=l1.fit_transform(data['cut'])
l1.classes_
data['cut_label']=label
l2=LabelEncoder()
label1=l2.fit_transform(data['clarity'])
data['clarity_label']=label1
data.head(2)
data['color']=data['color'].map({'D':1, 'E':2, 'F':3, 'G':4, 'H':5, 'I':6, 'J':7, 'NA':8})#.astype(int)
data['color'].fillna(0)
data['color']=data['color'].fillna(method='ffill')
y=data['price']
x=data.drop(['price','cut','clarity'],axis=1)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=42)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.fit_transform(x_test)

from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(n_estimators=50)
rf.fit(x_train,y_train)
pred2=rf.predict(x_test)

pickle.dump(rf,open('model.pkl','wb'))

model=pickle.load(open('model.pkl','rb'))
print(model.predict([[0.23,2,3,2]]))