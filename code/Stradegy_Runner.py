## 2022-2-20 luke
## Wrapper class for crossover runner

import numpy as np
import pandas as pd
from scipy import optimize
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
from LSTM_Predictor import LSTM_Predictor
import os

class Stradegy_Runner():
    """Realization of double avergae line stradegy"""
    def __init__(self,obs_length=30,pred_length=60,initial_wait=120,trade_cooldown=30,common_cooldown=15,train_interval=20,win_short=5,win_long=15):
        """Initialization"""
        self.trade = False
        self.present_date = datetime.strptime('09-11-2016','%m-%d-%Y')
        self.last_train_date = datetime.strptime('09-11-2016','%m-%d-%Y')
        self.last_trade_date = datetime.strptime('09-11-2016','%m-%d-%Y')
        self.end_date = datetime.strptime('09-10-2021','%m-%d-%Y')
        self.cross_start_date = datetime.strptime('09-11-2016','%m-%d-%Y')
        self.cross_end_date = datetime.strptime('09-11-2016','%m-%d-%Y')

        ## Assets
        self.cash = 1000.0
        self.gold = 0.0 # in troy ounce
        self.bitcoin = 0.0 # in bitcoin
        self.total_asset = 1000.0 # in dollar
        self.gold_price = 0
        self.bitcoin_price = 0
        self.gold_trade = 0
        self.bitcoin_trade = 0
        self.gold_commission = 0.01
        self.bitcoin_commission = 0.02

        ## Parameters
        self.obs_length = obs_length
        self.pred_length = pred_length
        self.initial_wait = initial_wait
        # self.trade_cooldown = trade_cooldown
        # self.common_cooldown  = common_cooldown
        # self.wait_time = 0
        self.train_interval = train_interval
        self.win_short = win_short 
        self.win_long = win_long 
        self.gold_epochs = 60
        self.bitcoin_epochs = 60
        ## Read data
        # self.gold_df = pd.read_csv("./2022_Problem_C_DATA/LBMA-GOLD.csv")
        # self.bitcoin_df = pd.read_csv("./2022_Problem_C_DATA/BCHAIN-MKPRU.csv")
        
        ## Trade hisory
        self.trade_history = []

        ## LSTM predictor
        self.Gold_predictor = LSTM_Predictor(label='gold')
        self.Bitcoin_predictor = LSTM_Predictor(label='bitcoin')
        self.gold_df, self.gold_date, self.gold_price = self.Gold_predictor.get_date_price()
        self.bitcoin_df, self.bitcoin_date, self.bitcoin_price = self.Bitcoin_predictor.get_date_price()

        ## 
        self.gold_next_date = datetime.strptime('09-11-2016','%m-%d-%Y')
        self.gold_next_price = 0.0
        self.bitcoin_next_date = datetime.strptime('09-11-2016','%m-%d-%Y')
        self.bitcoin_next_price = 0.0
        self.next_trade_date = datetime.strptime('09-11-2016','%m-%d-%Y')
        self.range = 5 # range for finding local max/min
        self.next_max_asset = -10000.0

        print(self.gold_df.Value[datetime.strptime('09-12-2016','%m-%d-%Y')])
        return

    def total_assets(self,print_info=False):
        self.total_asset = self.cash
        self.total_asset += self.gold * self.gold_price
        self.total_asset += self.bitcoin * self.bitcoin_price
        if print_info:
            print()
            print(self.present_date)
            print(f"Total assets: ${self.total_asset}")
            print(f"Cash: ${self.cash}")
            print(f"Gold: {self.gold}  ${self.gold * self.gold_price}")
            print(f"Bitcoin: {self.bitcoin}  ${self.bitcoin * self.bitcoin_price}\n")

        return self.total_asset

    def Trade(self):
        """Trade with current assets."""
        self.trade = False
        if self.gold_trade >= 10e-5:
            self.gold += self.gold_trade
            self.cash -= self.gold_trade * self.gold_price
            self.cash -= abs(self.gold_trade) * self.gold_price * self.gold_commission
            self.last_trade_date = self.present_date
            self.trade = True
        if self.bitcoin_trade >= 10e-5:
            self.bitcoin += self.bitcoin_trade
            self.cash -= self.bitcoin_trade * self.bitcoin_price
            self.cash -= abs(self.bitcoin_trade) * self.bitcoin_price * self.bitcoin_commission
            self.last_trade_date = self.present_date
            self.trade = True
        if self.trade:
            self.total_assets(print_info=False)
            self.trade_history.append([self.present_date, # 0
                                       self.gold_trade, # 1
                                       self.gold_trade*self.gold_price,  # 2
                                       self.bitcoin_trade, # 3
                                       self.bitcoin_trade*self.bitcoin_price, # 4
                                       self.cash, # 5
                                       self.gold, # 6
                                       self.gold*self.gold_price, # 7
                                       self.bitcoin, # 08
                                       self.bitcoin*self.bitcoin_price, # 9
                                       self.total_asset # 10
                                       ])
            print(f"\nTrade on {self.present_date}:")
            print(f"Gold price: ${self.gold_price}")
            print(f"Gold: {self.gold_trade}  ${self.gold_trade * self.gold_price}")
            print(f"Bitcoin price: ${self.bitcoin_price}")
            print(f"Bitcoin: {self.bitcoin_trade}  ${self.bitcoin_trade * self.bitcoin_price}\n")
        return self.total_asset

    def update_epochs(self):
        """Update epochs according to previous train loss."""
        ## Update bitcoin epoch
        bitcoin_loss = self.Bitcoin_predictor.get_loss()
        if bitcoin_loss > 0.01:
            self.bitcoin_epochs = 40
        elif bitcoin_loss > 0.005:
            self.bitcoin_epochs = 20
        else:
            self.bitcoin_epochs = 10
        ## Update gold epoch
        gold_loss = self.Gold_predictor.get_loss()
        if gold_loss > 0.045:
            self.gold_epochs = 60
        elif gold_loss > 0.03:
            self.gold_epochs = 30
        else:
            self.gold_epochs = 15

        return self.gold_epochs, self.bitcoin_epochs

    def average_judge(self):
        """Trading stradegy of measuring average predicted price"""
        self.trade = False
        self.gold_trade = 0.0
        self.bitcoin_trade = 0.0
        ## Gold Trade
        if self.gold_pred.mean() >= self.gold_price * 1.03:
            self.gold_trade = self.cash * 0.15 / self.gold_price
            self.trade = True
        elif self.gold_pred.mean() <= self.gold_price * 0.97:
            self.gold_trade = - self.gold * 0.3
            self.trade = True
        ## Bitcoin Trade
        if self.bitcoin_pred.mean() >= self.bitcoin_price * 1.05:
            self.bitcoin_trade = self.cash * 0.15 / self.bitcoin_price
            self.trade = True
        elif self.bitcoin_pred.mean() <= self.bitcoin_price * 0.95:
            self.bitcoin_trade = - self.bitcoin * 0.3
            self.trade = True

        return self.trade

    def peak_trough(self):
        """Stradegy of finding by locating price local peaks and troughs"""
        self.next_max_asset = -10000.0
        # print("\n####\n")
        ## Gold Prediction
        self.gold_found = False
        for i in range(self.range, len(self.gold_pred)-self.range):
            if self.gold_pred[i] == max(self.gold_pred[i-self.range:i+self.range]) or self.gold_pred[i] == min(self.gold_pred[i-self.range:i+self.range]):
                index = i + self.gold_date.index(self.present_date)
                self.gold_next_date = self.gold_date[index]
                self.gold_found = True
                break
        if not self.gold_found:
            index = self.gold_date.index(self.present_date)+len(self.gold_pred) - self.range
            self.gold_next_date = self.gold_date[index]
        
        ## Bitcoin Prediction
        self.bitcoin_found = False
        for i in range(self.range, len(self.bitcoin_pred)-self.range):
            if self.bitcoin_pred[i] == max(self.bitcoin_pred[i-self.range:i+self.range]) or self.bitcoin_pred[i] == min(self.bitcoin_pred[i-self.range:i+self.range]):
                index = i + self.bitcoin_date.index(self.present_date)
                self.bitcoin_next_date = self.bitcoin_date[index]
                self.bitcoin_found = True
                break
        if not self.bitcoin_found:
            index = self.bitcoin_date.index(self.present_date) + len(self.bitcoin_pred) - self.range
            self.bitcoin_next_date = self.bitcoin_date[index]

        self.next_trade_date = min(self.gold_next_date,self.bitcoin_next_date)
        while not self.next_trade_date in self.gold_date:
            self.next_trade_date += timedelta(days=1)

        self.gold_next_price = self.gold_df.Value[self.next_trade_date]
        self.bitcoin_next_price = self.bitcoin_df.Value[self.next_trade_date]
        ## Linear programming x1 - gold x2 - bitcoin
        ## ++ +- -+ --
        c = -np.array([[self.gold_next_price-self.gold_price*(1+self.gold_commission), self.bitcoin_next_price-self.bitcoin_price*(1+self.bitcoin_commission)],
                       [self.gold_next_price-self.gold_price*(1+self.gold_commission), self.bitcoin_next_price-self.bitcoin_price*(1-self.bitcoin_commission)],
                       [self.gold_next_price-self.gold_price*(1-self.gold_commission), self.bitcoin_next_price-self.bitcoin_price*(1+self.bitcoin_commission)],
                       [self.gold_next_price-self.gold_price*(1-self.gold_commission), self.bitcoin_next_price-self.bitcoin_price*(1-self.bitcoin_commission)]])
        A_ub = np.array([[[ self.gold_price*(1+self.gold_commission), self.bitcoin_price*(1+self.bitcoin_commission)]],
                         [[ self.gold_price*(1+self.gold_commission),-self.bitcoin_price*(1+self.bitcoin_commission)]],
                         [[-self.gold_price*(1+self.gold_commission), self.bitcoin_price*(1+self.bitcoin_commission)]],
                         [[-self.gold_price*(1+self.gold_commission),-self.bitcoin_price*(1+self.bitcoin_commission)]]])
        b_ub =  self.cash - 0.001
        x_b = [(0,self.cash/(1+self.gold_commission)/self.gold_price),(0,self.cash/(1+self.gold_commission)/self.gold_price),(-self.gold,-0),(-self.gold,-0)]
        y_b = [(0,self.cash/(1+self.bitcoin_commission)/self.bitcoin_price),(-self.bitcoin,-0),(0,self.cash/(1+self.bitcoin_commission)/self.bitcoin_price),(-self.bitcoin,-0)]
        for i in range(0,4):
            res = optimize.linprog(c=c[i], A_ub=A_ub[i], b_ub=b_ub, bounds=(x_b[i],y_b[i]),method='simplex')
            print(res.x)
            print(res.slack)
            print(self.cash-res.fun+self.gold_next_price*self.gold+self.bitcoin_next_price*self.bitcoin)
            if self.cash-res.fun+self.gold_next_price*self.gold+self.bitcoin_next_price*self.bitcoin > self.next_max_asset:
                self.next_max_asset = self.cash-res.fun+self.gold_next_price*self.gold+self.bitcoin_next_price*self.bitcoin
                self.gold_trade = res.x[0]
                self.bitcoin_trade = res.x[1]
        self.total_assets()
        self.trade = True
        # if self.next_max_asset > self.total_asset:
        #     self.trade = True
        # else:
        #     self.trade = False
        #     self.gold_trade = 0.0
        #     self.bitcoin_trade = 0.0

        self.present_date = self.next_trade_date

        return self.trade

    # def crossover(self):
    #     """Trading stradegy of double average line crossover"""
    #     self.trade = False
    #     self.gold_trade = 0.0
    #     self.bitcoin_trade = 0.0
    #     self.present_index = len(self.obs) - 1 - int(self.win_long/2)
    #     ## Gold
    #     self.gold_short_average = np.zeros_like()
    #     self.gold_long_average = np.zeros_like()
    #     self.gold_combined = np.hstack(self.gold_obs,self.gold_pred)

    def update_date(self):
        """Update present_date,cross_start_date and cross_end_date."""
        ## present_date
        # self.present_date = min(self.bitcoin_next_date, self.gold_next_date)
        # self.present_date += timedelta(days=self.wait_time)
        if self.present_date > self.end_date - timedelta(days=15):
            return
        # while not self.present_date in self.gold_date:
        #     self.present_date += timedelta(days=1)
        ## cross_start_date
        self.cross_start_date = self.present_date - timedelta(days=self.obs_length) # - int(self.win_long/2)
        while not ((self.cross_start_date in self.gold_date) and (self.present_date in self.bitcoin_date)):
            self.cross_start_date -= timedelta(days=1)
        ## cross_end_date
        self.cross_end_date = self.present_date + timedelta(days=self.pred_length) # + int(self.win_long/2)
        if self.cross_end_date >= self.end_date:
            self.cross_end_date = self.end_date
            return
        while not ((self.cross_end_date in self.gold_date) and (self.present_date in self.bitcoin_date)):
                self.cross_end_date += timedelta(days=1)
        return

    def run(self):
        """Run stradegy"""
        ## Initialize time
        self.present_date = datetime.strptime('09-11-2016','%m-%d-%Y') + timedelta(days=self.initial_wait)
        self.update_date()

        while self.present_date < self.end_date:
            ## Update price
            self.gold_price = self.gold_df.Value[self.present_date]
            self.bitcoin_price = self.bitcoin_df.Value[self.present_date]
            train = self.last_train_date + timedelta(days=self.train_interval) <=  self.present_date # whether to update model
            # train = False
            self.update_epochs()
            self.gold_obs,self.gold_pred = self.Gold_predictor.get_data(self.cross_start_date,
                                                                                        self.present_date,
                                                                                        self.cross_end_date,
                                                                                        train=train,
                                                                                        epochs=self.gold_epochs
                                                                                        )
            self.bitcoin_obs,self.bitcoin_pred = self.Bitcoin_predictor.get_data(self.cross_start_date,
                                                                                self.present_date,
                                                                                self.cross_end_date,
                                                                                train=train,
                                                                                epochs=self.bitcoin_epochs
                                                                                )
            if train:
                self.last_train_date = self.present_date
            ## Trade decision
            self.peak_trough()
            self.Trade()
            self.total_assets(print_info=True)
            ## Update date
            # if self.trade:
            #     self.wait_time = self.trade_cooldown
            # else:
            #     self.wait_time = self.common_cooldown
            self.update_date()

        ## Last day
        self.present_date = self.end_date
        self.total_assets(print_info=True)
        
        ## Print trade info
        self.trade_history = np.array(self.trade_history)
        plt.figure("Trade History")
        plt.title('Trade History')
        plt.plot(self.gold_date,self.gold_price,label = 'Gold Price')
        plt.plot(self.bitcoin_date,self.bitcoin_price,label = 'Bitcoin Price')
        plt.plot(self.trade.history[:,0],self.trade.history[:,2],label = 'Gold Trade')
        plt.plot(self.trade.history[:,0],self.trade.history[:,4],label = 'Bitcoin Trade')
        plt.plot(self.trade.history[:,0],self.trade.history[:,5],label = 'Cash')
        plt.plot(self.trade.history[:,0],self.trade.history[:,7],label = 'Gold')
        plt.plot(self.trade.history[:,0],self.trade.history[:,9],label = 'Bitcoin')
        plt.plot(self.trade.history[:,0],self.trade.history[:,10],label = 'Total Assets')
        plt.xlabel('Date')
        plt.xlabel('US Dollar')
        plt.legend()
        plt.savefig('./results/Trade_history.png')

        return
        



if __name__ == "__main__":
    my_runner = Stradegy_Runner()
    my_runner.run()