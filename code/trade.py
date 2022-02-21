## 2022-2-20 luke
## Run Trading with prediction and trading stradegies

from datetime import datetime, timedelta
from LSTM_Predictor import LSTM_Predictor
from Stradegy_Runner import Stradegy_Runner

if __name__ == "__main__":
    my_runner = Stradegy_Runner()
    my_runner.run()
