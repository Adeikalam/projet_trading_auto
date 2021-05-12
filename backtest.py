# -*- coding: utf-8 -*-
"""
Created on Wed May 12 17:44:39 2021

@author: Pierre
"""

from backtesting import Backtest, Strategy
from backtesting.lib import crossover

from backtesting.test import SMA, GOOG
from modeling import build_pipeline

pipeline = build_pipeline('scaler.joblib', 'model.joblib')


class Ma_strategie(Strategy):

    def init(self):
        self.model = pipeline

    def next(self):
        jour_en_cours = self.data.df.iloc[-1:]
        
        volatilite_estimee = self.model.predict(jour_en_cours)
        
        if volatilite_estimee:
            if jour_en_cours['Open'].iloc[0] > jour_en_cours['Open_1'].iloc[0]:
                self.buy()
            else:
                self.sell()



def backtest_model(data, model, cash, commission):
    
    bt = Backtest(data, Ma_strategie,
              cash=10000, commission=0,
              exclusive_orders=True)

    output = bt.run()
    fig = bt.plot(open_browser=False)
    
    return fig, output