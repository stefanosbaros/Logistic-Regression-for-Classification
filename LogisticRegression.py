#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 17:59:00 2020

@author: stefanosbaros
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: stefanosbaros
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score,mean_squared_error
from sklearn import metrics


# Our goal in this projest is to use logistic regression to predict
# whether someone with a certain profile of physiological values will get
# diabetes or not

# loading data

diabetes_path = '/Users/stefanosbaros/Desktop/diabetes.csv'

all_features = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']

diabetes_data = pd.read_csv(diabetes_path,  names=all_features)

feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']

X = diabetes_data[feature_cols] # features


y = diabetes_data['label'] # labels



# dividing data into train and test sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)


# define the model 
log_reg_model = LogisticRegression()

# fit the model with data
log_reg_model.fit(X_train,y_train)


y_pred=log_reg_model.predict(X_test)


# plotting Glucose vs label
plt.scatter(X_test.glucose,y_pred, color='b',linestyle='-',label='predicted')
plt.xlabel('Glucose')
plt.ylabel('Diabetes')
plt.title('Prediction')
plt.legend(loc='upper left');
plt.show()


# plotting blood pressure vs label
plt.scatter(X_test.bp,y_pred, color='b',linestyle='-',label='predicted')
plt.xlabel('Blood Pressure')
plt.ylabel('Diabetes')
plt.title('Prediction')
plt.legend(loc='upper left');
plt.show()

# plotting BMI vs label
plt.scatter(X_test.bmi,y_pred, color='b',linestyle='-',label='predicted')
plt.xlabel('BMI')
plt.ylabel('Diabetes')
plt.title('Prediction')
plt.legend(loc='upper left');
plt.show()


# Use score method to get accuracy of model
score = log_reg_model.score(X_test, y_test)
print(score)

# Printing confusion matrix
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)