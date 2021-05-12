# -*- coding: utf-8 -*-
"""
Created on Wed May 12 17:34:47 2021

@author: Pierre
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def preprocess_data(df):
    # Retours sur investissement
    R = [1, 5, 10, 15]
    
    for r in R:
        df['R_' + str(r)] = df['Open'].pct_change(r)
        
    # Moyennes Mobiles
    M = [5, 10, 15, 20]
    
    for m in M:
        df['MA_' + str(m)] = df['Open'].rolling(m).mean()
        
    # Historique des prix
    
    P = range(1, 11)
    
    for p in P:
        df['Open_' + str(p)] = df['Open'].shift(p)
    
    # Historique des volumes
    V = range(1, 11)
    
    for v in V:
        df['Volume_' + str(v)] = df['Volume'].shift(v)
    
    # Jour de la semaine
    df['Weekday'] = pd.to_datetime(df['Date']).dt.weekday
    
    # Volatilités passées
    
    V = [5, 10, 15]
    
    for v in V:
        df['V_' + str(v)] = df['R_1'].rolling(v).std()
        
    df['target'] = df['V_5'].shift(-5)
    df['target'] = df['target'].apply(lambda x : 1 if x > 0.011 else 0)
    df = df.dropna()
    
    return df