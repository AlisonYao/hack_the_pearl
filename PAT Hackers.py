#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
import numpy as np

#%matplotlib inline
#%load_ext autoreload
#%autoreload 2


#from HSPCAPI import load_transaction_history
#df = load_transaction_history()

from HSBCAPI import load_target_labels

df = load_target_labels()
lst = df.Purchase_201707.keys()
df.head()

#Keep previous purchasing behavior as a feature
Y_previous = (df.Purchase_201707 + df.Purchase_201708 + df.Purchase_201709 + df.Purchase_201710 + df.Purchase_201711 + df.Purchase_201712 + df.Purchase_201801 + df.Purchase_201802 + df.Purchase_201803)

#These are the ones we want to predict, red zone
Y_val      = ((df.Purchase_201804 + df.Purchase_201805)> 0)*1
Y_target   = (df.Purchase_201806)

plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
Y_previous.value_counts().plot(kind="bar",title="Previous purchases %.2f%%" % np.round(np.sum(Y_previous>0)/ Y_previous.shape[0]*100,2));
plt.subplot(1,3,2)
Y_val.value_counts().plot(kind="bar",title="Purchases to predict %.2f%%" % np.round(np.sum(Y_val>0)/ Y_val.shape[0]*100,2));
plt.subplot(1,3,3)
Y_target.value_counts().plot(kind="bar",title="Purchases to predict %.2f%%" % np.round(np.sum(Y_target>0)/ Y_target.shape[0]*100,2));

Y_target = (Y_target > 0)*1

from HSBCAPI import load_customer_info
dfX = load_customer_info()
dfX.head()

dfX = dfX[["age","Gender", "Salary", 'Number_Children','NationCode'
           ,'PBK_Ind','HIB_Status']]
dfXY =  dfX.merge(Y_previous.rename("prev_purchase"), on="Customer_id")
dfXY = dfXY.merge(Y_val.rename("label_train"),on="Customer_id")
dfXY = dfXY.merge(Y_target.rename("label_test"),on="Customer_id")
dfXY.head()

#cut 1
X_tr = dfXY[["age","Gender", "Salary", 'Number_Children', 'NationCode',\
             'PBK_Ind',"prev_purchase",'HIB_Status']]
X_te = dfXY[["age","Gender", "Salary", 'Number_Children', 'NationCode',\
             'PBK_Ind',"prev_purchase",'HIB_Status']]
Y_tr = dfXY[["label_train"]]
Y_te = dfXY[["label_test"]]

#cut 2
cut = int(np.round(dfXY.shape[0]*.8))
X_tr = X_tr[0:cut]
X_te = X_te[(cut+1):]

#our target
y_true_tr = Y_tr.label_train[0:cut]
y_true_te = Y_te.label_test[(cut+1):]

from sklearn.preprocessing import StandardScaler
from sklearn.compose       import ColumnTransformer
from sklearn.linear_model  import LogisticRegression
from sklearn.impute        import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline      import Pipeline

#PIPELINE 1: for numeric variables
numeric_features = ['age', "Salary", 'Number_Children'\
                    ,'PBK_Ind', 'prev_purchase']
numeric_transformer = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
        #...
        #..
    ]
)

#PIPELINE 2
categorical_features = ['Gender', 'NationCode','HIB_Status']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
    #...
    #...
])

#combine pipeline 1 & 2
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

#add a classifier ....
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression(penalty="l2"))])

#train and predict - usually you want to do some cross validation here
#note how the fit function only takes TRAIN data as input. The train data was cut both horizontally and vertically.
logreg_model = clf.fit(X_tr,y_true_tr)

#the predictions of the test data are based on just the test data
logreg_scores = logreg_model.predict_proba(X_te)
y_pred_lr = logreg_scores[:,1]

def auc(y_true, y_pred):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_true, y_pred)
print('LR average precision-recall score: {0:0.4f}'.format(auc(y_true_te, y_pred_lr)))

from sklearn.metrics import average_precision_score
#Average precision-recall score

print('LR average precision-recall score: {0:0.2f}'.format(average_precision_score(y_true_te, y_pred_lr)))


print('\nHere is the list of the IDs of the customers that are going to invest:')
for i in range(len(y_pred_lr)):
    if y_pred_lr[i] > 0.5:
        print(lst[i])
       


  
