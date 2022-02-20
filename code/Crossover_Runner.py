## 2022-2-20 luke
## Wrapper class for crossover runner

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
from LSTM_Predictor import LSTM_Predictor
import os

class Crossover_Runner():
    """Realization of double avergae line stradegy"""
    def __init__(self,obs_length=60,pred_length=60,initial_wait=120,trade_cooldown=100,common_cooldown=80,train_interval=20,win_short=3,win_long=20):
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
        self.trade_cooldown = trade_cooldown
        self.common_cooldown  = common_cooldown
        self.wait_time = 0
        self.train_interval = train_interval
        self.win_short = win_short 
        self.win_long = win_long 
        self.gold_epochs = 60
        self.bitcoin_epochs = 60
        ## Read data
        # self.gold_df = pd.read_csv("./2022_Problem_C_DATA/LBMA-GOLD.csv")
        # self.bitcoin_df = pd.read_csv("./2022_Problem_C_DATA/BCHAIN-MKPRU.csv")
        
        ## Trade hisory
        # self.trade = 

        ## LSTM predictor
        self.Gold_predictor = LSTM_Predictor(label='gold')
        self.Bitcoin_predictor = LSTM_Predictor(label='bitcoin')
        self.gold_date =  self.Gold_predictor.get_date()
        self.bitcoin_date =  self.Bitcoin_predictor.get_date()

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

    def update_date(self):
        """Update present_date,cross_start_date and cross_end_date."""
        ## present_date
        self.present_date += timedelta(days=self.wait_time)
        if self.present_date > self.end_date - timedelta(days=7):
            return
        while not ((self.present_date in self.gold_date) and (self.present_date in self.bitcoin_date)):
            self.present_date += timedelta(days=1)
        ## cross_start_date
        self.cross_start_date = self.present_date - timedelta(days=self.obs_length)
        while not ((self.cross_start_date in self.gold_date) and (self.present_date in self.bitcoin_date)):
            self.cross_start_date -= timedelta(days=1)
        ## cross_end_date
        self.cross_end_date = self.present_date + timedelta(days=self.pred_length)
        if self.cross_end_date >= self.end_date:
            self.cross_end_date = self.end_date
            return
        while not ((self.cross_end_date in self.gold_date) and (self.present_date in self.bitcoin_date)):
                self.cross_end_date += timedelta(days=1)
        return

    def Trade(self):
        """Trade with current assets."""
        if self.trade:
            self.gold += self.gold_trade
            self.bitcoin += self.bitcoin_trade
            self.cash -= self.gold_trade * self.gold_price
            self.cash -= self.bitcoin_trade * self.bitcoin_price
            ## Pay commission
            self.cash -= abs(self.gold_trade) * self.gold_price * self.gold_commission
            self.cash -= abs(self.bitcoin_trade) * self.bitcoin_price * self.bitcoin_commission
            # self.total_assets(print_info=False)
            self.last_trade_date = self.present_date
        return self.total_asset

    def update_epochs(self):
        """Update epochs according to previous train loss."""
        ## Update bitcoin epoch
        bitcoin_loss = self.Bitcoin_predictor.get_loss()
        if bitcoin_loss > 0.01:
            self.bitcoin_epochs = 60
        elif bitcoin_loss > 0.005:
            self.bitcoin_epochs = 30
        else:
            self.bitcoin_epochs = 16
        ## Update gold epoch
        gold_loss = self.Gold_predictor.get_loss()
        if gold_loss > 0.035:
            self.gold_epochs = 60
        elif gold_loss > 0.02:
            self.gold_epochs = 30
        else:
            self.gold_epochs = 16

        return self.gold_epochs, self.bitcoin_epochs

    def crossover(self):
        self.trade = False
        self.gold_trade = 0.0
        self.bitcoin_trade = 0.0
        ## Gold Trade
        if self.gold_pred.mean() >= self.gold_price * 1.03:
            self.gold_trade = self.cash * 0.1 / self.gold_price
            self.trade = True
        elif self.gold_pred.mean() <= self.gold_price * 0.97:
            self.gold_trade = - self.gold * 0.3
            self.trade = True
        ## Bitcoin Trade
        if self.bitcoin_pred.mean() >= self.bitcoin_price * 1.05:
            self.bitcoin_trade = self.cash * 0.1 / self.bitcoin_price
            self.trade = True
        elif self.bitcoin_pred.mean() <= self.bitcoin_price * 0.95:
            self.bitcoin_trade = - self.bitcoin * 0.3
            self.trade = True

        return self.trade

    def run(self):
        """Run stradegy"""
        ## Initialize time
        self.present_date = datetime.strptime('09-11-2016','%m-%d-%Y') + timedelta(days=self.initial_wait)
        self.update_date()

        while self.present_date < self.end_date:
            
            train = self.last_train_date + timedelta(days=self.train_interval) <=  self.present_date # whether to update model
            # train = False
            self.update_epochs()
            self.gold_obs,self.gold_pred,self.gold_price = self.Gold_predictor.get_data(self.cross_start_date,
                                                                                        self.present_date,
                                                                                        self.cross_end_date,
                                                                                        train=train,
                                                                                        epochs=self.gold_epochs
                                                                                        )
            self.bitcoin_obs,self.bitcoin_pred,self.bitcoin_price = self.Bitcoin_predictor.get_data(self.cross_start_date,
                                                                                self.present_date,
                                                                                self.cross_end_date,
                                                                                train=train,
                                                                                epochs=self.bitcoin_epochs
                                                                                )
            if train:
                self.last_train_date = self.present_date

            ## Trade decision
            self.trade = self.crossover()
            self.Trade()
            self.total_assets(print_info=True)

            if self.trade:
                self.wait_time = self.trade_cooldown
            else:
                self.wait_time = self.common_cooldown

            ## Update date
            self.update_date()

        ## Trade everything on last day
        self.present_date = self.end_date
        # self.trade = True
        # self.gold_tarde = - self.gold
        # self.bitoin_tarde = - self.bitcoin
        # self.Trade()
        self.total_assets(print_info=True)
        
        ## Print trade info


        return
        



if __name__ == "__main__":
    my_runner = Crossover_Runner()
    my_runner.run()