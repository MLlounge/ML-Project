# you can watch more on ' www.mllounge.com '
# LSTM model to predict stock

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import rmsprop, Adam
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class LstmModel():
    def __init__(self):
        self.data = pd.read_csv("./KODEX_200.csv", delimiter=',', dtype=int)    # read data from kodex_200.csv file
        self.x = self.data.iloc[:-1, :]                              # set data x. (d_day - 1)'s price.
                                                                     # [start price, highest price, lowest price, end price]
        self.x = np.asarray(self.x)                     # change to np type.
        self.x = np.reshape(self.x, [-1, 1, 4])
        self.x_max = np.max(self.x)
        self.x = self.x / self.x_max        # normalization

        self.y = self.data.iloc[1:, 3]                   # set data y. d_day's end price.
        self.y = np.asarray(self.y)                     # change to np type
        self.y = np.reshape(self.y, [-1, 1])
        self.y_max = np.max(self.y)
        self.y = self.y / self.y_max                # normalization

        self.learning_rate = 0.001
        self.model = self.build_model()


    def build_model(self):      # function to build model
        model = Sequential()
        model.add(LSTM(2048, input_shape=[1, 4], activation='tanh', return_sequences=True))  # LSTM
        model.add(LSTM(1024, activation='tanh', return_sequences=True))  # LSTM
        model.add(LSTM(512, activation='tanh'))  # LSTM
        # can input ' dropout=0.25, recurrent_dropout=0.25 ' on ' LSTM( ) '
        # EX) model.add(LSTM(512, activation='tanh', dropout=0.25, recurrent_dropout=0.25))

        model.add(Dense(256, activation='relu'))
        model.add(Dense(1, activation='linear'))            # output 1 value. linear value.

        model.summary()
        model.compile(rmsprop(lr=self.learning_rate), loss="mse")

        return model


    def train(self):
        self.model.fit(self.x, self.y, batch_size=128, epochs=200)    # train


    def predict(self, x):
        p_x = np.reshape(x, [1, 1, 4]) / self.x_max     # set x(input) data for prediction

        pre = self.model.predict(p_x) * self.y_max      # prediction
        return pre      # return predicted value


    def plt_chart(self):    # show plot chart

        x_time = np.arange(1, 599)

        plt.subplot(2, 1, 1)      # chart for real data
        plt.plot(x_time, np.reshape(self.y, [598]))

        plt.subplot(2, 1, 2)      # chart for predict data
        pre_y_for_plot = np.zeros([598])
        for i in range(598):
            pre_y_for_plot[i] = self.predict(self.x[i, 0, :])
        plt.plot(x_time, pre_y_for_plot)

        plt.show()


if __name__ == "__main__":
    lstm = LstmModel()
    lstm.train()  # start training

    # set data for prediction
    pre_data = pd.read_csv('./KODEX_200.csv', delimiter=',', dtype=int)
    pre_data = np.asarray(pre_data)
    pre_data = pre_data[-1, :]      # final day's x(input) data

    pre_y = lstm.predict(pre_data)  # prediction
    print("\npredicted value = ", pre_y[0, 0])
    print("real value = ", 31645)

    # show chart
    lstm.plt_chart()