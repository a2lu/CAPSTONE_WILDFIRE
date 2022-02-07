"""
Anthony Chi, Jasku Singh, Oscar Jimene, Alice Lu
baselines.py
"""
import sys
import json
import os
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt

import geopandas as gpd
import pandas as pd
from datetime import datetime, timedelta
import time
import numpy as np
import copy
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier

class baselines():
    """
    Contains functions for baseline modeling.
    """
    
    def log_reg(clean_df, X_scaled):
        """
        Description: takes in the standardized dataframe and array and fits a Logistic Regression model
        Parameters: df, array -> model
        Returns: fitted model
        """
        lab_enc = preprocessing.LabelEncoder()
        encoded = lab_enc.fit_transform(clean_df[['burnSeverity']].values)
        x_train, x_test, y_train, y_test = train_test_split(X_scaled, encoded,test_size=0.2)
        pipe = LogisticRegression()
        pipe.fit(x_train, y_train)
        print("Logistic Regression: "+str(pipe.score(x_test, y_test)))
        return pipe

    def mlp_class(clean_df, X_scaled):
        """
        Description: takes in the standardized dataframe and array and fits a MLP Classifier model
        Parameters: df, array -> model
        Returns: fitted model
        """
        lab_enc = preprocessing.LabelEncoder()
        encoded = lab_enc.fit_transform(clean_df[['burnSeverity']].values)
        x_train, x_test, y_train, y_test = train_test_split(X_scaled, encoded,test_size=0.2)
        clf = MLPClassifier(max_iter=300).fit(x_train, y_train)
        print("MLP Classifier: "+str(clf.score(x_test, y_test)))
        return clf


    def xtra_class(clean_df, X_scaled):
        """
        Description: takes in the standardized dataframe and array and fits a Extra Trees Classifier model
        Parameters: df, array -> model
        Returns: fitted model and confusion matrix
        """
        lab_enc = preprocessing.LabelEncoder()
        encoded = lab_enc.fit_transform(clean_df[['burnSeverity']].values)
        x_train, x_test, y_train, y_test = train_test_split(X_scaled, encoded,test_size=0.2)
        clf = ExtraTreesClassifier().fit(x_train, y_train)
        print("Extra Trees Classifier: "+str(clf.score(x_test, y_test)))
        y_pred = clf.predict(x_test)
        return clf, confusion_matrix(y_test, y_pred)

    def kn_class(clean_df, X_scaled):
        """
        Description: takes in the standardized dataframe and array and fits a KNeighbors Classifier model
        Parameters: df, array -> model
        Returns: fitted model
        """
        lab_enc = preprocessing.LabelEncoder()
        encoded = lab_enc.fit_transform(clean_df[['burnSeverity']].values)
        x_train, x_test, y_train, y_test = train_test_split(X_scaled, encoded,test_size=0.2)
        clf = KNeighborsClassifier()
        clf.fit(x_train, y_train)
        print("Extra Trees Classifier: "+str(clf.score(x_test, y_test)))
        return clf

    def rf_class(clean_df, X_scaled):
        """
        Description: takes in the standardized dataframe and array and fits a Random Forest Classifier model
        Parameters: df, array -> model
        Returns: fitted model
        """
        lab_enc = preprocessing.LabelEncoder()
        encoded = lab_enc.fit_transform(clean_df[['burnSeverity']].values)
        x_train, x_test, y_train, y_test = train_test_split(X_scaled, encoded,test_size=0.2)
        clf = RandomForestClassifier()
        clf.fit(x_train, y_train)
        print("Random Forest Classifier: "+str(clf.score(x_test, y_test)))
        return clf