# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 10:20:36 2021

@author: Artur
"""

import pandas as pd

base = pd.read_csv('D://Cursos//DataScience//Mach_Learn_e_Dt_Scie_com_Python//Datasets//credit_data.csv')

for x in base.index:
    if base.loc[x, 'age'] < 0:
        base.loc[x, 'age'] = base['age'][base.age > 0].median()
del(x)

base['age'].fillna(base['age'].median(), inplace=True)

X = base.iloc[:, 1:4]

Y = base.iloc[:, 4]

# from sklearn import preprocessing
# X = preprocessing.scale(X)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

from sklearn.naive_bayes import GaussianNB
cla = GaussianNB()

cla.fit(X_train, Y_train)
Y_pred = cla.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
precision = accuracy_score(Y_test, Y_pred)
c_matrix = confusion_matrix(Y_test, Y_pred)