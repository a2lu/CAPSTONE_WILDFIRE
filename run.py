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

sys.path.insert(0, 'src')
from src.data import *

def main(targets):
    """
    this method will run all the methods within class data_exploration.py
    """
    # Read in the datasets
    df = data.parse_data()

    # Clean the dataframe
    clean_df = data.clean_data(df)

    # Standardize each feature
    X_scaled = data.standardize(clean_df)

if __name__ == "__main__":
    targets = sys.argv[1:]
    main(targets)