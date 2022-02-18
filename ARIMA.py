# 2022-2-18 ARIMA time series prediction

import warnings
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Defaults
plt.rcParams['figure.figsize'] = (20.0, 10.0)
plt.rcParams.update({'font.size': 12})
plt.style.use('ggplot')

class ARIMA:
    def __init__(self,gold_path,bit_path):
        """initialization"""
        self.gold_path = gold_path
        self.bit_path = bit_path
        self.gold_data = pd.read_csv(gold_path, engine='python', skipfooter=3)
        self.bit_data = pd.read_csv(bit_path, engine='python', skipfooter=3)

    def plot_data(self):
        """plot the data from csv files"""
        # plot gold data
        plt.figure(1)
        self.gold_data.plot()
        plt.ylabel('Monthly airline passengers (x1000)')
        plt.xlabel('Date')

        # plot bit coin data
        plt.figure(2)
        self.bit_data.plot()
        plt.ylabel('Monthly airline passengers (x1000)')
        plt.xlabel('Date')
        plt.show()


    def predict(self,train_start,train_end,test_end):
        """predict preices using ARIMA"""
        result = {}
        return result