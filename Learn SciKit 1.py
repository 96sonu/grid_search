# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 11:25:20 2018

@author: doshi
"""

import pandas as pd
from numpy import array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.naive_bayes import GaussianNB as NB
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.ensemble import BaggingClassifier as BG
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import AdaBoostClassifier as ABC
from xgboost import XGBClassifier as XGB

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt'

df = pd.read_csv(url, header=None, names=['variance','skewness','curtosis','entropy','class'])
X = df[['variance','skewness','curtosis','entropy']]
y = df[['class']]

zero_class_x = []
one_class_x = []

zero_variance = []
one_variance = []

for x in range(0,1372):
    if(df.values[x][4] == 0.0):
        zero_class_x.append(df.iloc[[x]])
    else:
        one_class_x.append(df.iloc[[x]])
                                                                                               
#print(X)
#print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#print(type(X_train))
#print(type(y_train))

y_train = array(y_train).flatten()
y_test = array(y_test).flatten()

scaler = StandardScaler()

scaler.fit(X_train)

#print(type(X_test))

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#print(type(X_test))

DTC_tuned_parameter = [{'max_depth': [4,5,6,7],          'min_samples_split':[8,9,10,12,14], 
                        'min_samples_leaf':[2,3,4,5],    'min_impurity_decrease':[0.0001,0.0005,0.001,0.005,0.01],
                        'max_features':[2,3,4],          'max_leaf_nodes':[1300,1400,1500,1600],
                        'min_weight_fraction_leaf':[0,0.01]}]

MLP_tunned_parameter = [{'hidden_layer_sizes':[(5,5,5)],
                         'activation':['identity','logistic','tanh','relu'],
                         'alpha':[0.1,0.3],
                         'learning_rate':['constant','adaptive'],
                         'max_iter':[1000], 'tol':[1e-1,1e-4],
                         'momentum':[0.9,1.0], 'early_stopping':[False,True]}]

SVC_tuned_parameters = [{'kernel': ['rbf', 'linear', 'poly'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100],
                         'random_state': [2,5,10,15]}]

NB_tuned_parameter = [{'priors': [[0.5,0.5],[0.25,0.75],[0.3,0.7],[0.6,0.4]]}]

LR_tuned_parameter = [{'penalty': ['l2'], 'tol': [1e-4], 'C': [1.0],
                       'fit_intercept': [True,False], 'intercept_scaling': [1], 
                       'max_iter': [100], 'multi_class': ['ovr','multinomial','auto'],
                       'solver': ['newton-cg','lbfgs','sag','saga']}]

KNN_tuned_parameter = [{'n_neighbors': [3,5,7,10], 'algorithm': ['auto','ball_tree','kd_tree','brute'], 
                        'p': [1,2,3], 'weights': ['uniform', 'distance']}]

BG_tuned_parameter = [{'n_estimators': [10,15,16,17], 'max_samples': [9,10,11,12], 
                       'max_features': [1,2,3,4], 'random_state': [5,6,7,8,9]}]

RFC_tuned_parameter = [{'n_estimators': [10,15,20], 'max_depth': [5,10,15,20], 'max_features':[1,2,3,4],
                        'criterion': ['gini','entropy'], 'min_samples_split': [2,3,4,5,6], 
                        'min_samples_leaf': [1,2,3,4,5,6,7]}]

ABC_tuned_parameter = [{'base_estimator': [DTC()], 'n_estimators': [10,15,20], 'learning_rate': [1,2,3,4,5], 
                        'algorithm': ['SAMME', 'SAMME.R'], 'random_state': [4,8,12,16]}]

GBC_tuned_parameter = [{'loss': ['deviance'], 'learning_rate': [0.4,0.5,0.8],
                        'n_estimators': [25,50,100,125], 'max_features': [1,2] }]

XGB_tuned_parameter = [{'learning_rate': [0.1,0.5,1,2], 'n_estimators': [10,20,25,30], 'seed': [1,2], 
                        'min_child_weight': [1,5,10,20,40,100], 'max_delta_step':[1,5,10,20,40,100]}]


print('1. Decision Tree')
print('2. Neural Net')
print('3. Support Vector Machine')
print('4. Gaussuian Naive Bayes')
print('5. Logistic Regression')
print('6. K-Nearest Neighbors')
print('7. Bagging')
print('8. Random Forest')
print('9. AdaBoost Classifier')
print('10. Gradient Boosting Classifier')
print('11. XGBoost')
print('Please enter the respective number of classifier you want to use')

x = int(input())

scores = ['average_precision','recall','f1','accuracy']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()
    
    if(x == 1):
        clf = GridSearchCV(DTC(), DTC_tuned_parameter, cv=5, scoring='%s' %score)
    elif(x == 2):
        clf = GridSearchCV(MLP(), MLP_tunned_parameter, cv=5, scoring ='%s' %score)
    elif(x == 3):
        clf = GridSearchCV(SVC(), SVC_tuned_parameters, cv=5, scoring='%s' % score)
    elif(x == 4):
        clf = GridSearchCV(NB(), NB_tuned_parameter, cv=7, scoring ='%s' %score)
    elif(x == 5):
        clf = GridSearchCV(LR(), LR_tuned_parameter, cv=7, scoring ='%s' %score)
    elif(x == 6):
        clf = GridSearchCV(KNN(), KNN_tuned_parameter, cv=7, scoring ='%s' %score)
    elif(x == 7):
        clf = GridSearchCV(BG(), BG_tuned_parameter, cv=7, scoring ='%s' %score)
    elif(x == 8):
        clf = GridSearchCV(RFC(), RFC_tuned_parameter, cv=7, scoring ='%s' %score)
    elif(x == 9):    
        clf = GridSearchCV(ABC(), ABC_tuned_parameter, cv=7, scoring ='%s' %score)
    elif(x == 10):
        clf = GridSearchCV(GBC(), GBC_tuned_parameter, cv=7, scoring = '%s' %score)
    elif(x == 11):
        clf = GridSearchCV(XGB(), XGB_tuned_parameter, cv=7, scoring = '%s' %score)

    print("Check Point")
    
    clf.fit(X_train, y_train)


    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
#    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#        print("%0.3f (+/-%0.03f) for %r"
#              % (mean, std * 2, params))
    print()
    
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print("Detailed confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print()
    if(score == 'average_precision'):
        print("Average Precision:")
        print(average_precision_score(y_true, y_pred))
    elif(score == 'recall'):
        print("Recall Score")
        print(recall_score(y_true,y_pred))
    elif(score == 'f1'):
        print("Average F1")
        print(f1_score(y_true,y_pred))
    elif(score == 'accuracy'):
        print("Accuracy Score:")
        print(accuracy_score(y_true, y_pred))    
    

    
