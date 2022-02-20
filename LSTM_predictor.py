from logging import raiseExceptions
from pickle import FALSE
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn.preprocessing
import sklearn.model_selection
import tensorflow as tf
from datetime import datetime,timedelta
import os

class LSTM_Predictor():
    """LSTM + CNN model wrapper for gold or bitcoin price forecast"""
    def __init__(self,label='gold',alpha=7,beta=2,gamma=64):
        """Initialization"""
        print("\nInitializing LSTM predictor for \n" + label.title() + ".")
        self.label = label
        self.path = './results/' + label + '/'
        if not os.path.exists('./results/'):
            os.mkdir('./results/')
        if not os.path.exists(self.path):
            os.mkdir(self.path)

        ## Model parameters
        self.alpha = alpha   # window length
        self.beta = beta   # the number of LSTM layers
        self.gamma = gamma  # the number of filters in convolutional layer
        self.diff = True

        ## Load data
        if label == 'gold':
            self.df = pd.read_csv("/Users/luke/Desktop/美赛/MCM-ICM-2022/2022_Problem_C_DATA/LBMA-GOLD.csv")
            prices = self.df['USD (PM)'].tolist()
        elif label == 'bitcoin':
            self.df = pd.read_csv("/Users/luke/Desktop/美赛/MCM-ICM-2022/2022_Problem_C_DATA/BCHAIN-MKPRU.csv")
            prices = self.df['Value'].tolist()
        else:
            raiseExceptions("Wrong label!")
        date = pd.to_datetime(self.df['Date']).tolist()

        self.prices = []
        self.date = []
        for i in range(0,len(prices)):
            if str(prices[i]) != 'nan': # delete nan
                self.prices.append(prices[i])
                self.date.append(date[i])
        # self.prices_diff = [self.prices[0]] # diff price
        # for i in range(1,len(self.prices)):
        #     self.prices_diff.append(self.prices[i]-self.prices[i-1])

        # plot data
        self.df.plot()
        plt.xlabel('Date')
        plt.ylabel(label.title() + ' Daily Price')
        plt.savefig(self.path + label.title() + '_Daily_Price.png')
        # plt.show()

        print("\nLoad data success!\n")

        self.scaler = sklearn.preprocessing.StandardScaler()

        self.start_date = datetime.strptime('01-11-2017','%m-%d-%Y')
        self.present_date = datetime.strptime('01-11-2019','%m-%d-%Y')
        self.end_date = datetime.strptime('01-11-2019','%m-%d-%Y')

        return

    def build_model(self,alpha=7,beta=1,gamma=64):
        ## Build model
        # Model parameters
        self.alpha = alpha   # window length
        self.beta = beta   # the number of LSTM layers
        self.gamma = gamma  # the number of filters in convolutional layer
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(self.alpha, input_shape=(1, self.alpha), return_sequences=True) for _ in range(self.beta)
        ] + [
            # attention.Attention(),
            tf.keras.layers.Dense(self.alpha, activation='linear'),
            tf.keras.layers.Reshape((1, self.alpha)),
            tf.keras.layers.Conv1D(filters=self.gamma, kernel_size=1, strides=1),
            tf.keras.layers.MaxPooling1D(pool_size=2, strides=1, padding='same'),
            tf.keras.layers.Dense(self.alpha, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear'),
        ])
        self.model.summary()
        return self.model

    def train_model(self,train_end_date,epochs=100,batch_size=64,diff=False):
        """Create dataset, train LSTM model and test the model"""
        self.diff = diff
        ## Create train & test data
        # Split data
        try:
            self.n_train = self.date.index(train_end_date)
            # self.n_test = self.date.index(test_end_date)
        except:
            raiseExceptions('Invalid Date!')
        self.train_end_date = train_end_date

        train_prices = np.array(self.prices[:self.n_train]).reshape((-1, 1))

        # Standardize data
        self.scaler.fit(train_prices)
        train_prices = self.scaler.transform(train_prices).reshape(-1)
        print("train_prices.shape:", train_prices.shape)
        self.train_prices = train_prices

        # Create train dataset
        self.x_train = []
        self.y_train = []
        for i in range(len(train_prices) - self.alpha):
            self.x_train.append(train_prices[i:i+self.alpha].reshape((1, -1)))
            if self.diff:
                # self.y_train.append(train_prices[i+self.alpha].reshape((1, -1))-train_prices[i+self.alpha-1].reshape((1, -1)))  # next day's price
                self.y_train.append(train_prices[i+self.alpha].reshape((1, -1)))
            else:
                self.y_train.append(train_prices[i+self.alpha].reshape((1, -1)))  # next day's price
        self.x_train = np.array(self.x_train, dtype='float32')
        self.y_train = np.array(self.y_train, dtype='float32')
        print("self.x_train.shape:", self.x_train.shape)
        print("self.y_train.shape:", self.y_train.shape)

        # Plot train date
        # plt.figure(figsize=(16, 4))
        # plt.subplot(121)
        # plt.title("self.x_train[:, 0, 0] (%d ~ %d)" % (0, len(self.x_train)-1))
        # sns.lineplot(x=np.arange(0, len(self.x_train)), y=self.x_train[:, 0, 0])
        # plt.subplot(122)
        # plt.title("X_test[:, 0, 0] (%d ~ %d)" % (len(self.x_train), len(self.x_train)+len(self.x_test)-1))
        # sns.lineplot(x=np.arange(len(self.x_train), len(self.x_train)+len(self.x_test)), y=self.x_test[:, 0, 0])
        # plt.show()

        ## train model
        self.model.compile(optimizer='adam', loss='mse')
        self.history = self.model.fit(self.x_train, self.y_train, batch_size=batch_size, epochs=epochs)
        plt.figure(figsize=(16, 8))
        sns.lineplot(data=self.history.history)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(train_end_date.strftime("%m-%d-%Y") + '_' + self.label.title() + '_train_history')
        plt.savefig(self.path + train_end_date.strftime("%m-%d-%Y") + '_' + '_train_history.png')
        # plt.show()
        return self.history

    def predict(self,test_end_date='12-19-2017'):
        """Predict result"""
        try:
            self.n_test = self.date.index(test_end_date)
        except:
            raiseExceptions('Invalid Date!')
        if self.n_train >= self.n_test:
            raiseExceptions('Invalid Date!')
        self.test_end_date = test_end_date

        # Split Data
        test_prices = np.array(self.prices[self.n_train:self.n_test]).reshape((-1, 1))
        test_prices = self.scaler.transform(test_prices).reshape(-1)
        print("test_prices.shape:", test_prices.shape)

        # Create test dataset
        self.x_test = []
        self.y_test = []
        for i in range(len(test_prices)-self.alpha):
            self.x_test.append(test_prices[i:i+self.alpha].reshape((1, -1)))
            self.y_test.append(test_prices[i+self.alpha].reshape((1, -1))) 
                
        self.x_test = np.array(self.x_test, dtype='float32')
        self.y_test = np.array(self.y_test, dtype='float32')
        print("self.x_test.shape:", self.x_test.shape)
        print("self.y_test.shape:", self.y_test.shape)

        ## Predict       
        self.preds = self.model.predict(self.x_test)
        # # preds = self.scaler.inverse_transform(preds.reshape(-1, 1)).reshape(-1)
        # start = 0
        # for i in range(0,len(self.train_prices_diff)):
        #     start += self.train_prices_diff[i]
        # if self.diff:
        #     self.preds[0] += start
        #     for i in range(1,len(self.preds)):
        #         self.preds[i] += self.preds[i-1]
        ## plot prediction
        plt.figure(figsize=(16, 8))
        sns.lineplot(data={
            "actual data": self.scaler.inverse_transform(self.y_test.reshape(-1, 1)).reshape(-1),
            "prediction": self.scaler.inverse_transform(self.preds.reshape(-1, 1)).reshape(-1),
        })
        plt.xlabel("Time")
        plt.ylabel(self.label.title() + " Daily Price")
        plt.title(self.train_end_date.strftime("%m-%d-%Y") + '_' +  self.test_end_date.strftime("%m-%d-%Y") + '_' + self.label.title() + '_Predictions')
        plt.savefig(self.path + self.train_end_date.strftime("%m-%d-%Y") + '_' +  self.test_end_date.strftime("%m-%d-%Y") + '_' + '_Predictions.png')
        # plt.show()
        return self.preds

    def get_data(self,start_date,present_date,end_date):
        """Get data for trading stradegy"""
        self.start_date = start_date
        self.present_date = present_date
        self.end_date = end_date
        try:
            self.n_start = self.date.index(start_date)
        except:
            raiseExceptions('Invalid Date!')

        self.build_model(alpha=7,beta=1,gamma=64)
        self.train_model(train_end_date = present_date + timedelta(days=1)) # today's price is known
        self.predict(test_end_date=end_date)

        self.observation = self.prices[self.n_start:self.n_train]
        self.prediction = self.prices[self.n_start:self.n_train]

        return self.observation, self.prediction, self.prices[self.n_train - 1]

        


if __name__ == "__main__":
    train_end_date = datetime.strptime('01-11-2019','%m-%d-%Y')
    test_end_date = datetime.strptime('02-22-2019','%m-%d-%Y')
    ## gold
    Gold_predictor = LSTM_Predictor(label='gold')
    Gold_predictor.build_model(alpha=7,beta=1,gamma=64)
    Gold_predictor.train_model(train_end_date=train_end_date)
    Gold_predictor.predict(test_end_date=test_end_date)

    ## bitcoin
    # Bitcoin_predictor = LSTM_Predictor(label='bitcoin')
    # Bitcoin_predictor.build_model(alpha=7,beta=1,gamma=64)
    # Bitcoin_predictor.train_model(train_end_date='9-11-17')
    # Bitcoin_predictor.predict(test_end_date='12-19-17')