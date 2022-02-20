# 2022-2-19 luke
# Run Trading with prediction and trading stradegies

from LSTM_Predictor import LSTM_Predictor

def Crossover_runner(obvervation,prediction,present_spot,short=3,long=20):
    """Realization of double avergae line stradegy"""
    def __init__(self):
        """Initialization"""

    def run(self):
        """Run stradegy"""


if __name__ == "__main__":
    ## gold
    Gold_predictor = LSTM_Predictor(label='gold')
    Gold_predictor.build_model(alpha=7,beta=1,gamma=64)
    Gold_predictor.train_model(train_end_date='7-11-17')
    Gold_predictor.predict(test_end_date='7-31-17')

    ## bitcoin
    Bitcoin_predictor = LSTM_Predictor(label='bitcoin')
    Bitcoin_predictor.build_model(alpha=7,beta=1,gamma=64)
    Bitcoin_predictor.train_model(train_end_date='9-11-17')
    Bitcoin_predictor.predict(test_end_date='12-19-17')