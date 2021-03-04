# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 11:32:18 2021

@author: Artur
"""

import pandas as pd
bcensus = pd.read_csv('D://Cursos\DataScience//Mach_Learn_e_Dt_Scie_com_Python//Datasets//census.csv')

X = bcensus.iloc[:, 0:14]

Y = bcensus.iloc[:, 14]

dummy_sex = pd.get_dummies(X['sex'])
dummy_workclass = pd.get_dummies(X['workclass'])
dummy_education = pd.get_dummies(X['education'])
dummy_maritalstatus = pd.get_dummies(X['marital-status'])
dummy_occupation = pd.get_dummies(X['occupation'])
dummy_relationship = pd.get_dummies(X['relationship'])
dummy_race = pd.get_dummies(X['race'])
dummy_nativecountry = pd.get_dummies(X['native-country'])

X_dummy = X.drop(['sex', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'native-country'], axis = 1)

col_X_dummy = X_dummy.columns


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in [1, 3, 5, 6, 7, 8, 9, 13]:
    X.iloc[:, i] = le.fit_transform(X.iloc[:, i])

from sklearn import preprocessing
X_dummy = preprocessing.scale(X_dummy)
X_dummy = pd.DataFrame(X_dummy, columns = col_X_dummy)
X_dummy = pd.concat([X_dummy, dummy_sex, dummy_workclass, dummy_education, dummy_maritalstatus, dummy_occupation, dummy_relationship, dummy_race, dummy_nativecountry], axis = 1)


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X_dummy,Y,test_size=0.25)


# 5° Passo:
from sklearn.naive_bayes import GaussianNB
cla = GaussianNB()
cla.fit(X_train, Y_train)
Y_pred = cla.predict(X_test)

# 6° Passo: 
from sklearn.metrics import confusion_matrix, accuracy_score
precision = accuracy_score(Y_test, Y_pred)
c_matrix = confusion_matrix(Y_test, Y_pred)
