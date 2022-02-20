# 2022-2-19 luke
# Run Trading with prediction and trading stradegies

from datetime import datetime,timedelta
from LSTM_Predictor import LSTM_Predictor

class Crossover_runner():
    """Realization of double avergae line stradegy"""
    def __init__(self,obs_length=30,pred_length=30,initial_wait=90,win_short=3,win_long=20):
        """Initialization"""
        self.present_date = datetime.strptime('09-11-2016','%m-%d-%Y')
        self.trade = False

        # Assets
        self.dollar = 1000.0
        self.gold = 0.0 # in troy ounce
        self.bitcoin = 0.0 # in bitcoin
        self.total_asset = 1000.0 # in dollar
        self.gold_price = 0
        self.bitcoin_price = 0
        # Parameters
        self.obs_length = obs_length
        self.pred_length = pred_length
        self.initial_wait = initial_wait
        self.trade_cooldown = 10
        self.common_cooldown  = 10
        self.wait_time = 10
        self.win_short = win_short 
        self.win_long = win_long 

        # LSTM predictor
        self.Gold_predictor = LSTM_Predictor(label='gold')
        self.Bitcoin_predictor = LSTM_Predictor(label='bitcoin')
        return

    def total_assets(self):
        self.total_assets = self.dollar
        self.total_assets += self.gold * self.gold_price
        self.total_assets += self.bitcoin * self.bitcoin_price
        return self.total_assets

    def run(self):
        """Run stradegy"""
        self.present_date = datetime.strptime('09-11-2016','%m-%d-%Y')
        self.present_date += timedelta(days=self.initial_wait)
        while self.present_date < datetime.strptime('09-10-2021','%m-%d-%Y'):
            self.trade = False
            # self.gold_obs,self.gold_pred = self.Gold_predictor.get_data(self.present_date - timedelta(days=self.obs_length),
            #                                                             self.present_date,
            #                                                             self.present_date + timedelta(days=self.pred_length)
            #                                                             )
            # self.bitcoin_obs,self.bitcoin_pred,self.bitcoin_price = self.Bitcoin_predictor.get_data(self.present_date - timedelta(days=self.obs_length),
            #                                                                     self.present_date,
            #                                                                     self.present_date + timedelta(days=self.pred_length)
            #                                                                     )
            print(self.present_date)
            if self.trade:
                self.wait_time = self.trade_cooldown
            else:
                self.wait_time = self.common_cooldown
            self.present_date += timedelta(days=self.wait_time)



if __name__ == "__main__":
    my_runner = Crossover_runner()
    my_runner.run()