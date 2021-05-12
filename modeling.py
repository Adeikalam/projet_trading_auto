# -*- coding: utf-8 -*-
from joblib import load, dump

from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline


def drop_columns(df):
    to_drop = ['Date', 'High', 'Low', 'Close', 'Volume', 'OpenInt']
    
    return df.drop(to_drop, axis = 1)


def build_pipeline(scaler_path, model_path):
    
    scaler = load(scaler_path)
    model = load(model_path)
    
    col_dropper = FunctionTransformer(drop_columns)

    pipeline = Pipeline([
        ("drop_columns", col_dropper),
        ("scaling", scaler),
        ("model", model)
    ])   
    
    return pipeline