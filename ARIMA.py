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
    def __init__(self,label='gold',datapath = "./2022_Problem_C_DATA/LBMA-GOLD.csv"):
        """initialization"""
        self.label = label
        if label == 'gold':
            self.datapath = "./2022_Problem_C_DATA/LBMA-GOLD.csv"
        elif label == 'bitcoin':
            self.datapath = "./2022_Problem_C_DATA/BCHAIN-MKPRU.csv"
        else:
            print('\nWrong label!\n')
            return
        self.data = pd.read_csv(self.datapath, engine='python', skipfooter=3)
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data.set_index(['Date'], inplace=True)
        self.data_diff1 = self.data.diff(1)
        self.data_diff2 = self.data_diff1.diff(1)
        print("\nLoad data success!\n")
        print(self.data)


    def plot_data(self,plot_fig=False):
        """plot the data from csv files"""
        print("\nplotting data\n")
        # plot data
        self.data.plot()
        plt.ylabel(self.label.title() + ' daily prices')
        plt.savefig('./results/' + self.label.title() + '_daily_prices.png')
        plt.xlabel('Date')

        if plot_fig:
            plt.show()


    def predict(self,train_start='2016-09-12',train_end='2020-09-07',test_end='2020-11-06',order=0,plot_fig=False):
        """predicting prices using ARIMA"""
        # prepare data
        if order == 0:
            train_data = self.data[train_start:train_end]
        elif order == 1:
            train_data = self.data_diff1[train_start:train_end]
        elif order == 2:
            train_data = self.data_diff2[train_start:train_end]

        # Define the d and q parameters to take any value between 0 and 1
        q = d = range(0, 2)
        # Define the p parameters to take any value between 0 and 3
        p = range(0, 4) # 4
        # Generate all different combinations of p, q and q triplets
        pdq = list(itertools.product(p, d, q))
        # pdq = pdq[0:1]
        # Generate all different combinations of seasonal p, q and q triplets
        seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

        # print('Examples of parameter combinations for Seasonal ARIMA...')
        # print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
        # print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
        # print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
        # print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
        # print("\n")

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
                    results = mod.fit(disp=-1)
                    print('SARIMAX{}x{} - AIC:{}'.format(param, param_seasonal, results.aic), end='\r')
                    AIC.append(results.aic)
                    SARIMAX_model.append([param, param_seasonal])
                except:
                    continue
        print('The smallest AIC is {} for model SARIMAX{}x{}'.format(min(AIC), SARIMAX_model[AIC.index(min(AIC))][0],SARIMAX_model[AIC.index(min(AIC))][1]))
        mod = sm.tsa.statespace.SARIMAX(train_data,
                                order=SARIMAX_model[AIC.index(min(AIC))][0],
                                seasonal_order=SARIMAX_model[AIC.index(min(AIC))][1],
                                enforce_stationarity=False,
                                enforce_invertibility=False)
        results = mod.fit(disp=-1)    
        print(results.summary())
        results.plot_diagnostics(figsize=(20, 14))

        # visualize predictions
        pred0 = results.get_prediction(start='2019-09-08', dynamic=False)
        pred0_ci = pred0.conf_int()
        pred1 = results.get_prediction(start='2019-09-08', dynamic=True)
        pred1_ci = pred1.conf_int()
        pred2 = results.get_forecast(test_end)
        pred2_ci = pred2.conf_int()
        ax = self.data.plot(figsize=(20, 16))
        # 恢复差分
        if order == 1:

        elif order == 2:

        pred0.predicted_mean.plot(ax=ax, label='1-step-ahead Forecast (get_predictions, dynamic=False)')
        pred1.predicted_mean.plot(ax=ax, label='Dynamic Forecast (get_predictions, dynamic=True)')
        pred2.predicted_mean.plot(ax=ax, label='Dynamic Forecast (get_forecast)')
        ax.fill_between(pred2_ci.index, pred2_ci.iloc[:, 0], pred2_ci.iloc[:, 1], color='k', alpha=.1)
        plt.ylabel('Monthly airline passengers (x1000)')
        plt.xlabel('Date')
        plt.legend()
        plt.savefig('./results/prediction_'+ self.label +'.png')
        if plot_fig:
            plt.show()
        return results




if __name__ == "__main__":
    # gold_model = ARIMA(label='gold')
    # gold_model.plot_data()
    # gold_model.predict()

    bitcoin_model = ARIMA(label='bitcoin')
    bitcoin_model.plot_data()
    bitcoin_model.predict()
    