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
        # print(self.data)


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


    def predict(self,train_start='2016-09-12',train_end='2019-09-07',test_end='2019-12-06',order=0,plot_fig=False):
        """predicting prices using ARIMA"""
        # prepare data
        if order == 0:
            train_data = self.data[train_start:train_end]
        elif order == 1:
            train_data = self.data_diff1[train_start:train_end]
        elif order == 2:
            train_data = self.data_diff2[train_start:train_end]
        print(train_data)
        # Define the d and q parameters to take any value between 0 and 1
        q = d = range(0, 3)
        # Define the p parameters to take any value between 0 and 3
        p = range(0, 4) # 4
        # Generate all different combinations of p, q and q triplets
        pdq = list(itertools.product(p, d, q))

        warnings.filterwarnings("ignore") # specify to ignore warning messages
        AIC = []
        ARIMA_model = []
        for param in pdq:
            try:
                mod = sm.tsa.arima.ARIMA(train_data,
                                        order=param,
                                        enforce_stationarity=True,
                                        enforce_invertibility=True)
                results = mod.fit()
                print('ARIMA{} - AIC:{}'.format(param, results.aic), end='\r')
                AIC.append(results.aic)
                ARIMA_model.append([param])
            except:
                continue
        print('The smallest AIC is {} for model ARIMA{}'.format(min(AIC), ARIMA_model[AIC.index(min(AIC))][0]))
        
        mod = sm.tsa.arima.ARIMA(train_data,
                                order=ARIMA_model[AIC.index(min(AIC))][0],
                                enforce_stationarity=True,
                                enforce_invertibility=True)
        results = mod.fit()
        print(results.summary())
        # results.plot_diagnostics(figsize=(20, 14))

        # visualize predictions
        pred0 = results.get_prediction(start='2019-10-08', dynamic=True)
        pred0_ci = pred0.conf_int()
        # pred1 = results.get_prediction(start='2019-09-08', dynamic=True)
        # pred1_ci = pred1.conf_int()
        # pred2 = results.get_forecast(test_end)
        # pred2_ci = pred2.conf_int()
        ax = self.data.plot(figsize=(20, 16))
        print("\n**************\n")
        # 恢复差分
        # if order == 0:
        #     prediction = pred2.predicted_mean
        # if order == 1:
        #     prediction = pred2.predicted_mean.cumsum() + self.data['Date'].index(train_start-1)
        #     prediction = prediction.dropna()
        # elif order == 2:
        print(pred0.predicted_mean.to_list())
        pred0.predicted_mean.plot(ax=ax, label='Dynamic Forecast (get_forecast)')
        # prediction.plot(ax=ax, label='Dynamic Forecast (get_forecast)')
        # ax.fill_between(pred2_ci.index, pred2_ci.iloc[:, 0], pred2_ci.iloc[:, 1], color='k', alpha=.1)
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
    bitcoin_model.predict(order=0)
    