#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 13:02:27 2020

@author: jaytamakuwala
"""

P(y=1) = e^z/1+e^z #### Binary logistic regression model
## Credit classification
import pandas as pd
import numpy as np
##3 importing the dataset
credit_df = pd.read_csv("German Credit Data.csv") 
## checking the status of credit score
credit_df.status.value_counts()
#
X_features = list(credit_df.columns)
X_features.remove('status')
## encoding into categorical features
encoded_credit_df = pd.get_dummies(credit_df[X_features], drop_first = True)
##using ols method same as lm for regression parameters
import statsmodels.api as sm
Y=credit_df.status
X=sm.add_constant(encoded_credit_df)
### splitting the dataset into train and test
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test= train_test_split(X,Y, test_size = 0.3, random_state = 0)
## buliding the logistic regression model by using logit to estimate the parameters
import statsmodels.api as sm
logit = sm.Logit(Y_train, X_train)
logit_model = logit.fit()
##
logit_model.summary2()
####only 8 features are significant,p=0.05
## taking only the signifcant values only
def get_significant_vars(lm):
    ##storing the p-values and corresponding columns names in a dataframe
    var_p_vals_df = pd.DataFrame(lm.pvalues)
    var_p_vals_df['vars']=var_p_vals_df.index
    var_p_vals_df.columns= ['pvals', 'vars']
    ## filter the column names where p-value is less than 0.05
    return list(var_p_vals_df[var_p_vals_df.pvals <=0.05]['vars'] )

significant_vars = get_significant_vars(logit_model)   
significant_vars
##building final model
final_logit = sm.Logit(Y_train, sm.add_constant(X_train[significant_vars])).fit()
final_logit.summary2() 
### predicting the test data
y_pred_df =pd.DataFrame({"actual":Y_test, "predicted_prob":final_logit.predict(sm.add_constant(X_test[significant_vars]))})
##making cut-off probability 
y_pred_df['predicted'] = y_pred_df.predicted_prob.map(lambda x:1 if x > 0.5 else 0)
import matplotlib.pyplot as plt
import seaborn as sn
## creatung confusion matrix by seaborn
def draw_cm(actual, predicted):
    from sklearn.metrics import confusion_matrix
    cm =confusion_matrix(actual, predicted, [1,0])
    sn.heatmap(cm, annot =True, fmt = '.2f', xticklabels = ["bad credit", "good credit"], yticklabels = ["bad credit", "good credit"])
    plt.ylabel('True label')
    plt.xlabel('predicted label')
    plt.show()
### to disply cm
draw_cm(y_pred_df.actual, y_pred_df.predicted) 
### measuring accuracies
from sklearn.metrics import classification_report
cr=classification_report(y_pred_df.actual, y_pred_df.predicted) 
## roc_auc score
from sklearn.metrics import roc_auc_score
roc_auc =roc_auc_score(y_pred_df.actual, y_pred_df.predicted)
roc_auc
