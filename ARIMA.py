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
    def __init__(self,gold_path,bitcoin_path):
        """initialization"""
        self.gold_path = gold_path
        self.bitcoin_path = bitcoin_path
        self.gold_data = pd.read_csv(gold_path, engine='python', skipfooter=3)
        self.bitcoin_data = pd.read_csv(bitcoin_path, engine='python', skipfooter=3)
        # self.gold_data['Date']=pd.to_datetime(self.gold_data['Date'], format='%m/%d/%Y')
        self.gold_data.set_index(['Date'], inplace=True)
        # self.bitcoin_data['Date']=pd.to_datetime(self.bitcoin_data['Date'], format='%m/%d/%Y')
        self.bitcoin_data.set_index(['Date'], inplace=True)
        print("\nLoad data success!\n")
        print(self.gold_data)


    def plot_data(self):
        """plot the data from csv files"""
        print("\nplotting data\n")
        # plot gold data
        self.gold_data.plot()
        plt.ylabel('Gold daily prices')
        plt.xlabel('Date')

        # plot bit coin data
        self.bitcoin_data.plot()
        plt.ylabel('Bitcoin daily prices')
        plt.xlabel('Date')
        plt.show()


    def predict(self,label = 'gold',train_start='9/12/16',train_end='9/12/18'):# 9/12/16
        """predict preices using ARIMA"""
        # prepare data
        if label == 'gold':
            # train_data = self.gold_data[train_start:train_end]
            train_data = self.gold_data[train_start:train_end]
        elif label == 'bitcoin':
            train_data = self.bitcoin_data['9/12/16':'9/12/18']
        else:
            print('\nWrong label!\n')
            return

        # Define the d and q parameters to take any value between 0 and 1
        q = d = range(0, 2)
        # Define the p parameters to take any value between 0 and 3
        p = range(0, 4)
        # Generate all different combinations of p, q and q triplets
        pdq = list(itertools.product(p, d, q))
        # Generate all different combinations of seasonal p, q and q triplets
        seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

        print('Examples of parameter combinations for Seasonal ARIMA...')
        print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
        print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
        print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
        print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

        warnings.filterwarnings("ignore") # specify to ignore warning messages
        AIC = []
        SARIMAX_model = []
        for param in pdq:
            for param_seasonal in seasonal_pdq:
                try:
                    mod = sm.tsa.statespace.SARIMAX(train_data,
                                                    order=param,
                                                    seasonal_order=param_seasonal,
                                                    enforce_stationarity=False,
                                                    enforce_invertibility=False)

                    results = mod.fit()

                    print('SARIMAX{}x{} - AIC:{}'.format(param, param_seasonal, results.aic), end='\r')
                    AIC.append(results.aic)
                    SARIMAX_model.append([param, param_seasonal])
                except:
                    continue
        print('The smallest AIC is {} for model SARIMAX{}x{}'.format(min(AIC), SARIMAX_model[AIC.index(min(AIC))][0],SARIMAX_model[AIC.index(min(AIC))][1]))
        results.plot_diagnostics(figsize=(20, 14))
        plt.show()
        return result




if __name__ == "__main__":
    arima_model = ARIMA(gold_path="./2022_Problem_C_DATA/LBMA-GOLD.csv",
                        bitcoin_path="./2022_Problem_C_DATA/BCHAIN-MKPRU.csv")
    # arima_model.plot_data()
    arima_model.predict(label='gold')