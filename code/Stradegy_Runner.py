## 2022-2-20 luke
## Wrapper class for crossover runner

import numpy as np
import pandas as pd
from scipy import optimize
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from LSTM_Predictor import LSTM_Predictor
import os

class Stradegy_Runner():
    """Realization of double avergae line stradegy"""
    def __init__(self,obs_length=20,pred_length=60,initial_wait=30,train_interval=20):
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
        self.gold_price = 0.0
        self.bitcoin_price = 0.0
        self.gold_trade = 0.0
        self.bitcoin_trade = 0.0
        self.gold_commission = 0.08
        self.bitcoin_commission = 0.16

        ## Parameters
        self.obs_length = obs_length
        self.pred_length = pred_length
        self.initial_wait = initial_wait
        self.train_interval = train_interval
        self.gold_epochs = 60
        self.bitcoin_epochs = 40

        ## LSTM predictor
        self.Gold_predictor = LSTM_Predictor(label='gold')
        self.Bitcoin_predictor = LSTM_Predictor(label='bitcoin')
        self.gold_df, self.gold_date, self.gold_prices = self.Gold_predictor.get_date_price()
        self.bitcoin_df, self.bitcoin_date, self.bitcoin_prices = self.Bitcoin_predictor.get_date_price()

        ## programming parameters
        self.gold_next_date = datetime.strptime('09-11-2016','%m-%d-%Y')
        self.gold_next_price = 0.0
        self.bitcoin_next_date = datetime.strptime('09-11-2016','%m-%d-%Y')
        self.bitcoin_next_price = 0.0
        self.next_trade_date = datetime.strptime('09-11-2016','%m-%d-%Y')
        self.range = 4 # range for finding local max/min
        self.next_max_asset = -10000.0
        self.trade_history = []
        self.trade_history.append([ self.present_date, # 0
                                    self.gold_trade, # 1
                                    self.gold_trade*self.gold_price,  # 2
                                    self.bitcoin_trade, # 3
                                    self.bitcoin_trade*self.bitcoin_price, # 4
                                    self.cash, # 5
                                    self.gold, # 6
                                    self.gold*self.gold_price, # 7
                                    self.bitcoin, # 8
                                    self.bitcoin*self.bitcoin_price, # 9
                                    self.total_asset # 10
                                    ])

        return

    def total_assets(self,print_info=False):
        self.total_asset = self.cash
        self.total_asset += self.gold * self.gold_price
        self.total_asset += self.bitcoin * self.bitcoin_price
        if print_info:
            print(f"\nTotal assets: ${self.total_asset}")
            print(f"Cash: ${self.cash}")
            print(f"Gold: {self.gold}  ${self.gold * self.gold_price}")
            print(f"Bitcoin: {self.bitcoin}  ${self.bitcoin * self.bitcoin_price}\n")

        return self.total_asset

    def Trade(self):
        """Trade with current assets."""
        self.trade = False
        if self.gold_trade >= 1e-19:
            self.gold += self.gold_trade
            self.cash -= self.gold_trade * self.gold_price
            self.cash -= abs(self.gold_trade) * self.gold_price * self.gold_commission
            self.last_trade_date = self.present_date
            self.trade = True
        if self.bitcoin_trade >= 1e-19:
            self.bitcoin += self.bitcoin_trade
            self.cash -= self.bitcoin_trade * self.bitcoin_price
            self.cash -= abs(self.bitcoin_trade) * self.bitcoin_price * self.bitcoin_commission
            self.last_trade_date = self.present_date
            self.trade = True
        if self.trade:
            self.total_assets(print_info=False)
            print(f"\nTrade on {self.present_date}:")
            print(f"Gold price: ${self.gold_price}")
            print(f"Gold: {self.gold_trade}  ${self.gold_trade * self.gold_price}")
            print(f"Bitcoin price: ${self.bitcoin_price}")
            print(f"Bitcoin: {self.bitcoin_trade}  ${self.bitcoin_trade * self.bitcoin_price}\n")
        self.trade_history.append([ self.present_date, # 0
                                    self.gold_trade, # 1
                                    self.gold_trade*self.gold_price,  # 2
                                    self.bitcoin_trade, # 3
                                    self.bitcoin_trade*self.bitcoin_price, # 4
                                    self.cash, # 5
                                    self.gold, # 6
                                    self.gold*self.gold_price, # 7
                                    self.bitcoin, # 8
                                    self.bitcoin*self.bitcoin_price, # 9
                                    self.total_asset # 10
                                    ])
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


    def find_next_date(self):
        """Stradegy of finding next trade date."""
        # print("\n####\n")
        self.gold_price = self.gold_df.Value[self.present_date]
        self.bitcoin_price = self.bitcoin_df.Value[self.present_date]
        ## Gold Prediction
        self.gold_found = False
        # for i in range(self.range, len(self.gold_pred)-self.range):
            # if self.gold_pred[i] == max(self.gold_pred[i-self.range:i+self.range]) or self.gold_pred[i] == min(self.gold_pred[i-self.range:i+self.range]):
        # for i in range(1, len(self.gold_pred)):
        #     if not self.gold_pred[i]>=self.gold_pred[i-1] == self.gold_pred[i+1]>=self.gold_pred[i]:
            # index = i + self.gold_date.index(self.present_date) + 1
            # self.gold_next_date = self.gold_date[index]
            # self.gold_found = True
        if max(self.gold_pred) + min(self.gold_pred) >= 2.0*self.gold_price:
            index = self.gold_pred.index(max(self.gold_pred)) + self.gold_date.index(self.present_date) + 1
        else:
            index = self.gold_pred.index(min(self.gold_pred)) + self.gold_date.index(self.present_date) + 1
        self.gold_next_date = self.gold_date[index]
        self.gold_found = True

        if not self.gold_found:
            index = self.gold_date.index(self.present_date) + len(self.gold_pred) - self.range
            self.gold_next_date = self.gold_date[index]
        
        ## Bitcoin Prediction
        self.bitcoin_found = False
        # for i in range(self.range, len(self.bitcoin_pred)-self.range):
        #     if self.bitcoin_pred[i] == max(self.bitcoin_pred[i-self.range:i+self.range]) or self.bitcoin_pred[i] == min(self.bitcoin_pred[i-self.range:i+self.range]):
        # for i in range(1, len(self.bitcoin_pred)):
        #     if not self.bitcoin_pred[i]>=self.bitcoin_pred[i-1] == self.bitcoin_pred[i+1]>=self.bitcoin_pred[i]:
        #         index = i + self.bitcoin_date.index(self.present_date) + 1
        #         self.bitcoin_next_date = self.bitcoin_date[index]
        #         self.bitcoin_found = True
        #         break
        if max(self.bitcoin_pred) + min(self.bitcoin_pred) >= 2.0*self.bitcoin_price:
            index = self.bitcoin_pred.index(max(self.bitcoin_pred)) + self.bitcoin_date.index(self.present_date) + 1
        else:
            index = self.bitcoin_pred.index(min(self.bitcoin_pred)) + self.bitcoin_date.index(self.present_date) + 1
        self.bitcoin_next_date = self.bitcoin_date[index]
        self.bitcoin_found = True
        if not self.bitcoin_found:
            index = self.bitcoin_date.index(self.present_date) + len(self.bitcoin_pred) - self.range
            self.bitcoin_next_date = self.bitcoin_date[index]
        
        ## Update prediction price on next trade date
        g_n_i_o = self.gold_date.index(self.gold_next_date) - self.gold_date.index(self.present_date) - 1
        self.gold_next_price = self.gold_pred[g_n_i_o]
        b_n_i_o = self.bitcoin_date.index(self.bitcoin_next_date) - self.bitcoin_date.index(self.present_date) - 1
        self.bitcoin_next_price =self.bitcoin_pred[b_n_i_o]
        
        g_i_s = self.gold_date.index(self.present_date) + 1 - len(self.gold_obs)
        g_i_e = self.gold_date.index(self.present_date) + 1 + len(self.gold_pred)
        b_i_s = self.bitcoin_date.index(self.present_date) + 1 - len(self.bitcoin_obs)
        b_i_e = self.bitcoin_date.index(self.present_date) + 1 + len(self.bitcoin_pred)
        plt.figure(num=f"{self.present_date} Predictions",figsize=(12,9))
        #  = plt.subplots()
        plt.title(f"{self.present_date} Predictions")
        ax_g = plt.gca()
        ax_g.set_ylabel('Gold Daily Price(USD)')
        ax_g.set_xlabel('Date')
        ax_g.plot(self.gold_date[g_i_s:g_i_e],np.hstack((self.gold_obs,self.gold_pred)),'tab:red',label = 'Gold Price')
        if self.gold_found:
            ax_g.plot(self.gold_next_date,self.gold_next_price,'tab:red',marker='o')
        ax_g.plot(self.present_date,self.gold_price,'tab:green',marker='o')
        ax_g.legend()
        ax_b = ax_g.twinx()
        ax_b.set_ylabel('Bitcoin Daily Price(USD)')
        ax_b.plot(self.bitcoin_date[b_i_s:b_i_e],np.hstack((self.bitcoin_obs,self.bitcoin_pred)),'tab:blue',label = 'Bitcoin Price')
        if self.bitcoin_found:
            ax_b.plot(self.bitcoin_next_date,self.bitcoin_next_price,'tab:blue',marker='o')
        ax_b.plot(self.present_date,self.bitcoin_price,'tab:green',marker='o')
        ax_b.legend()
        # plt.axvline(x=self.present_date)
        # fig.tight_layout()
        plt.savefig(f"./results/{self.present_date}_Predictions")
        plt.close()

        # self.next_trade_date = min(self.gold_next_date, self.bitcoin_next_date)
        self.next_trade_date = min(self.gold_next_date, self.bitcoin_next_date)
        while (not self.next_trade_date in self.gold_date) or self.next_trade_date <= self.present_date:
            self.next_trade_date += timedelta(days=1)
        return

    def Linear_Programing(self):
        """Finding trade amount by linear programming"""
        ## (x1-gold x2-bitcoin) ++ +- -+ --
        self.next_max_asset = 0.0
        c = -np.array([[self.gold_next_price-self.gold_price*(1+self.gold_commission), self.bitcoin_next_price-self.bitcoin_price*(1+self.bitcoin_commission)],
                       [self.gold_next_price-self.gold_price*(1+self.gold_commission), self.bitcoin_next_price-self.bitcoin_price*(1-self.bitcoin_commission)],
                       [self.gold_next_price-self.gold_price*(1-self.gold_commission), self.bitcoin_next_price-self.bitcoin_price*(1+self.bitcoin_commission)],
                       [self.gold_next_price-self.gold_price*(1-self.gold_commission), self.bitcoin_next_price-self.bitcoin_price*(1-self.bitcoin_commission)]])
        A_ub = np.array([[[ self.gold_price*(1+self.gold_commission), self.bitcoin_price*(1+self.bitcoin_commission)]],
                         [[ self.gold_price*(1+self.gold_commission),-self.bitcoin_price*(1+self.bitcoin_commission)]],
                         [[-self.gold_price*(1+self.gold_commission), self.bitcoin_price*(1+self.bitcoin_commission)]],
                         [[-self.gold_price*(1+self.gold_commission),-self.bitcoin_price*(1+self.bitcoin_commission)]]])
        b_ub =  self.cash - 0.1
        # x_b = [(0,self.cash/(1+self.gold_commission)/self.gold_price),(0,self.cash/(1+self.gold_commission)/self.gold_price),(-self.gold,-0),(-self.gold,-0)]
        # y_b = [(0,self.cash/(1+self.bitcoin_commission)/self.bitcoin_price),(-self.bitcoin,-0),(0,self.cash/(1+self.bitcoin_commission)/self.bitcoin_price),(-self.bitcoin,-0)]
        x_b = [(0,None),(0,None),(-self.gold,0),(-self.gold,0)]
        y_b = [(0,None),(-self.bitcoin,0),(0,None),(-self.bitcoin,0)]

        for i in range(0,4):
            res = optimize.linprog(c=c[i], A_ub=A_ub[i], b_ub=b_ub, bounds=(x_b[i],y_b[i]),method='simplex')
            print(res)
            # print(res.slack)
            # print(self.cash-res.fun+self.gold_next_price*self.gold+self.bitcoin_next_price*self.bitcoin)
            if self.cash-res.fun+self.gold_next_price*self.gold+self.bitcoin_next_price*self.bitcoin > self.next_max_asset:
                self.next_max_asset = self.cash-res.fun+self.gold_next_price*self.gold+self.bitcoin_next_price*self.bitcoin
                self.gold_trade = res.x[0]
                self.bitcoin_trade = res.x[1]
        self.total_assets(print_info=False)

        self.present_date = self.next_trade_date
        return

    def update_date(self):
        """Update cross_start_date and cross_end_date."""

        ## start_date
        self.cross_start_date = self.present_date - timedelta(days=self.obs_length) # - int(self.win_long/2)
        while not ((self.cross_start_date in self.gold_date) and (self.present_date in self.bitcoin_date)):
            self.cross_start_date -= timedelta(days=1)
        ## end_date
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

        while self.present_date < self.end_date - timedelta(days=self.range + 1):
        # while False:
            print(f"\n\n{self.present_date}")
            ## Update price
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
            self.find_next_date()
            self.Linear_Programing()
            self.Trade()
            self.total_assets(print_info=True)
            # if self.present_date > self.end_date - timedelta(days=7):
            #     break
            self.update_date()

        ## Last day
        self.present_date = self.end_date
        self.total_assets(print_info=True)
        
        ## Print trade info
        self.trade_history = np.array(self.trade_history)
        plt.figure(num="Trade History",figsize=(12,9))
        plt.title('Trade History')
        # plt.plot(self.gold_date,self.gold_prices*0.1,label = 'Gold Price * 0.1')
        # plt.plot(self.bitcoin_date,self.bitcoin_prices*0.1,label = 'Bitcoin Price * 0.1')
        # plt.plot(self.trade_history[:,0],self.trade_history[:,2],label = 'Gold Trade',marker='o')
        # plt.plot(self.trade_history[:,0],self.trade_history[:,4],label = 'Bitcoin Trade',marker='o')
        # plt.plot(self.trade_history[:,0],self.trade_history[:,5],label = 'Cash')
        # plt.plot(self.trade_history[:,0],self.trade_history[:,7],label = 'Gold')
        # plt.plot(self.trade_history[:,0],self.trade_history[:,9],label = 'Bitcoin')
        plt.plot(self.trade_history[:,0],self.trade_history[:,10],label = 'Total Assets')
        plt.xlabel('Date')
        plt.ylabel('US Dollar')
        plt.legend()
        plt.savefig('./results/Trade_history.png')
        # print(self.trade_history)
        return
        


if __name__ == "__main__":
    my_runner = Stradegy_Runner()
    my_runner.run()