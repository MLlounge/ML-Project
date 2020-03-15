# you can watch more on ' www.mllounge.com '
# DNN model to learn ln(x)
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt


class Data:     # Create data
    def __init__(self):
        self.y = 0      # y value.
        self.x = np.arange(1, 301) * 3  # create x for training.
                                        # 3 ~ 900 -> [3, 6, 9, 12, ...]
        self.e = np.e       # e value.

    def calc_e(self, x):    # function to return ln(x).
                            # you can change the mathematical expression.
        return np.log(x) / np.log(self.e)   # return ln(x).

    def ret_xy(self):   # return x and y(= ln(x))
        self.y = self.calc_e(self.x)    # calcurate ln(x) and input to y.
        return self.x, self.y           # return x and y


class Dnn:  # create, train, predict DNN model.
            # make plot chart for trained data.
    def __init__(self):
        data = Data()
        self.x, self.y = data.ret_xy()  # set x, y value for training.
        self.learning_rate = 0.003
        self.input_shape = 1
        self.model = self.build_model() # create moel.

    def build_model(self):  # function for building model.
        model = Sequential()
        model.add(Dense(256, input_dim=self.input_shape, activation='relu'))    # relu
        #model.add(Dropout(0.25))
        model.add(Dense(128, activation='relu'))    # relu
        #model.add(Dropout(0.25))
        model.add(Dense(64, activation='relu'))     # relu
        model.add(Dense(1, activation='linear'))    # linear

        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))    # mse

        return model


    def train(self):    # function for training.
        self.model.fit(np.reshape(self.x, [-1, 1]), np.reshape(self.y, [-1, 1]), epochs=1000)


    def predict(self, pre_x):   # return predict value of DNN model. input data = pre_x.
        return self.model.predict(np.reshape(pre_x, [1, 1]))


    def plot_chart(self):   # make plot chart for trained data.
        plt.subplot(2, 1, 1)
        plt.plot(self.x, self.y, '-')



if __name__ == '__main__':  # main
    dnn = Dnn()
    dnn.train()     # train the model.

    # predict ln(x) with untrained data by using DNN model.
    p = [13, 70, 1000, 1250, 2000, 3000]  # input 13, 70, 1000, 1250, 2000 = untrained data x.
    for i in range(len(p)):
        pre = dnn.predict(p[i])     # predict.

        # pre_ans = set the real(answer) value of predict value.
        pre_ans = np.log(np.full((1, 300), p[i]))/np.log(np.e)

        # print predict and real value.
        print('\npredict(', p[i], ') = ', pre)
        print('real(', p[i], ') = ', pre_ans[0, 1])

        # print the difference between predict and real value.
        print('difference(', p[i], ') = ', pre - pre_ans[0, 1])


    # show plot chart.
    # plot chart for trained values.
    dnn.plot_chart()

    # plot chart for predict values.
    pre_x = np.arange(1, 301) * 3
    pre_y = np.zeros(300)
    for i in range(300):
        pre_y[i] = dnn.predict(pre_x[i])
    plt.subplot(2, 1, 2)
    plt.plot(pre_x, pre_y, '--')

    plt.show()