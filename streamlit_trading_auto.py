# -*- coding: utf-8 -*-
"""
Created on Wed May 12 17:35:26 2021

@author: Pierre
"""

import streamlit as st
import pandas as pd
from preprocessing import preprocess_data
from modeling import build_pipeline
from backtest import backtest_model


st.title("Projet Trading Automatique")

st.markdown("""
            Ce projet a été réalisé pendant la masterclass ML1 - Cadrer un projet de machine learning avec
            la promotion AVR21 BC DADS
            
            
            Le dataset utilisé se trouve [ici.](https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs)            
            """)
            
            
df = pd.read_csv("Data/Stocks/ibm.us.txt")

df = preprocess_data(df)

X = df.drop('target', axis = 1)
y = df['target']

X_train = X.iloc[:int(len(X)*0.7)]
y_train = y.iloc[:int(len(y)*0.7)]

X_test = X.iloc[int(len(X)*0.7):]
y_test = y.iloc[int(len(y)*0.7):]

pipeline = build_pipeline('scaler.joblib', 'model.joblib')

fig, output= backtest_model(X_test, pipeline, cash = 10000, commission = 0)

st.bokeh_chart(fig)

