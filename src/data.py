"""
Anthony Chi, Jasku Singh, Oscar Jimene, Alice Lu
data.py
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

class data():
    """
    class data contains functions for data cleaning and baseline modeling.
    """

    def parse_data():
        """
        Description: reads a csv file and converts into dataframe
        Parameters: none
        Returns: DataFrame
        """
        burned = gpd.read_file("../data/burned/sampleData.csv")
        unburned = gpd.read_file("../data/unburned/sampleData.csv")
        temp1 = gpd.GeoDataFrame()
        temp2 = gpd.GeoDataFrame()
        for i in np.unique(burned['FIRE_NAME']):
            temp1 = pd.concat([temp1,burned[burned['FIRE_NAME']==i]])
            temp2 = pd.concat([temp2,unburned[unburned['FIRE_NAME']==i]])
        df = pd.concat([temp1,temp2])
        return df

    def clean_data(df):
        """
        Description: reads a dataframe and cleans it
        Parameters: df -> clean_df
        Returns: DataFrame
        """
        clean_df = df[['burnSeverity','SR_B3','SR_B6','NDVI','elevation','percent_tree_cover','x_coord','y_coord']]
        for i in clean_df.columns:
            clean_df[i]=pd.to_numeric(clean_df[i], errors='coerce')
        for i in clean_df.columns:
            clean_df[i].fillna(value=clean_df[i].mean(), inplace=True)
        return clean_df

    def standardize(clean_df):
        """
        Description: reads a clean dataframe and standardize the features
        Parameters: clean_df -> array
        Returns: array
        """
        X_train = clean_df.drop(['burnSeverity'],axis=1)
        scaler = preprocessing.StandardScaler().fit(X_train.values)
        X_scaled = scaler.transform(X_train)
        return X_scaled

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
        Returns: fitted model
        """
        lab_enc = preprocessing.LabelEncoder()
        encoded = lab_enc.fit_transform(clean_df[['burnSeverity']].values)
        x_train, x_test, y_train, y_test = train_test_split(X_scaled, encoded,test_size=0.2)
        clf = ExtraTreesClassifier().fit(x_train, y_train)
        print("Extra Trees Classifier: "+str(clf.score(x_test, y_test)))
        return clf

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
        print("Extra Trees Classifier: "+str(clf.score(x_test, y_test)))
        return clf