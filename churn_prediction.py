# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 18:15:07 2019

@author: ramsw
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
df = pd.read_csv('Churn_rate.csv')

#First 10 values of the dataset
df.head(10)

#To check if any missing columns in the dataset
df_isnull = df.isnull().values

#To print the unique values in the dataset
df_isunique = df.nunique()
print(df_isunique)


df['TotalCharges'] = df["TotalCharges"].replace(" ",np.nan)
df["TotalCharges"] = df["TotalCharges"].astype(float)
df['TotalCharges'] = df['TotalCharges'].fillna((df['TotalCharges'].mean()))


#MultipleLines,OnlineSecurity,OnlineBackup,DeviceProtection,
#TechSupport,StreamingTV,StreamingMovies have 3 categories. 
#No service is equal to No.

df['MultipleLines'] = df['MultipleLines'].replace(['No phone service'], 'No')
df['OnlineSecurity'] = df['OnlineSecurity'].replace(['No internet service'], 'No')
df['OnlineBackup'] = df['OnlineBackup'].replace(['No internet service'], 'No')
df['DeviceProtection'] = df['DeviceProtection'].replace(['No internet service'], 'No')
df['TechSupport'] = df['TechSupport'].replace(['No internet service'], 'No')
df['StreamingTV'] = df['StreamingTV'].replace(['No internet service'], 'No')
df['StreamingMovies'] = df['StreamingMovies'].replace(['No internet service'], 'No')


#Using the map function to map Yes and No values to 1 and 0. 
d = {'Yes':1 , 'No': 0}
df['Churn'].map(d)
df['OnlineSecurity'].map(d)
df['OnlineBackup'].map(d)  
df['DeviceProtection'].map(d)
df['TechSupport'].map(d)
df['StreamingTV'].map(d)
df['StreamingMovies'].map(d)
df['MultipleLines'].map(d)

#Encoding the columns with 2 values

from sklearn.preprocessing import LabelEncoder
labelencoder_df = LabelEncoder()
Col1 = df.nunique()[df.nunique() == 2].keys().tolist()
for i in Col1:
    df[i]=labelencoder_df.fit_transform(df[i])

#Encoding the columns with multiple values

df['InternetService'] = pd.Categorical(df['InternetService'])
df_cat1 = pd.get_dummies(df['InternetService'], prefix = 'category')

df['Contract'] = pd.Categorical(df['Contract'])
df_cat2 = pd.get_dummies(df['Contract'], prefix = 'category')

df['PaymentMethod'] = pd.Categorical(df['PaymentMethod'])
df_cat3 = pd.get_dummies(df['PaymentMethod'], prefix = 'category')

df = pd.concat([df,df_cat1,df_cat2,df_cat3], axis = 1)

df = df.drop('InternetService', 1)
df = df.drop('Contract', 1)
df = df.drop('PaymentMethod', 1)


#Tenure column can be divided into bins 

def tenure_bin(df):
    if df["tenure"] <= 12 :
        return "Tenure1"
    elif (df["tenure"] > 13) & (df["tenure"] <= 24 ):
        return "Tenure2"
    elif (df["tenure"] > 25) & (df["tenure"] <= 36) :
        return "Tenure3"
    elif (df["tenure"] > 37) & (df["tenure"] <= 48) :
        return "Tenure4"
    elif df["tenure"] > 60 :
        return "Tenure5"
    
df["tenure_bin"] = df.apply(lambda df:tenure_bin(df), axis = 1)

df['tenure_bin'] = pd.Categorical(df['tenure_bin'])
df_cat4 = pd.get_dummies(df['tenure_bin'], prefix = 'category')

df = pd.concat([df,df_cat4], axis = 1)
df = df.drop('tenure_bin',1)
df = df.drop('tenure',1)


#Scaling the numerical columns
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

num_col1 = ['MonthlyCharges', 'TotalCharges']
df_num1 = df[num_col1].values
df_num1_scaled = sc.fit_transform(df_num1)
df_temp = pd.DataFrame(df_num1_scaled, columns = num_col1, index = df.index)

df_temp = df_temp.rename(columns={'MonthlyCharges': 'MonthlyCharges_scaled', 'TotalCharges': 'TotalCharges_scaled'})
df = pd.concat([df,df_temp], axis = 1)
df = df.drop('MonthlyCharges',1)
df = df.drop('TotalCharges',1)
df = df.drop('customerID',1)
#Dividing the columns into dependent and independent variable.
#Churn is dependent variable.

y = df.iloc[:, 13].values
x= df.drop('Churn',1)

#Splitting the data into Test and Train
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

#Fitting the model 
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(x_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)






