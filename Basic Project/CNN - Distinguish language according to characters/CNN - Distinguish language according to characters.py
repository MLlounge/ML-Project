# you can watch more on ' www.mllounge.com '
# Basic CNN project

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense
from keras.optimizers import Adam, rmsprop
import numpy as np
import cv2
import serial
from PIL import Image
import time
import sys


class CnnModel():
    def __init__(self):
        self.x_train, self.x_test, self.y_train, self.y_test = np.load("./image/data.npy")    # read data from data.npy file.
        self.x_train = np.reshape(self.x_train, [-1, 32, 32, 1])    # reshape x_train to 32 * 32 * 1
        self.x_test = np.reshape(self.x_test, [-1, 32, 32, 1])      # reshape x_test to 32 * 32 * 1

        self.input_shape = self.x_train.shape[1:]
        self.learning_rate = 0.001
        self.model = self.build_model()


    def build_model(self):      # bulid CNN model
        model = Sequential()
        model.add(Conv2D(64, kernel_size=(3, 3), padding='SAME', activation='relu', input_shape=self.input_shape))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, kernel_size=(3, 3), padding='SAME', activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(3, activation='softmax'))   # softmax.

        model.summary()

        model.compile(optimizer=rmsprop(lr=self.learning_rate), loss='categorical_crossentropy',
                      metrics=['accuracy'])  # rmsprop optimizer.
#        model.compile(optimizer=Adam(lr=self.learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])   # adam optimizer.
        return model


    def train(self):
        self.model.fit(self.x_train, self.y_train, batch_size=128, epochs=50)  # training.


    def predict(self, pre_img):
        pre = self.model.predict(pre_img)   # prediction.
        pre = pre.argmax()  # ont-hot to integer number.

        if pre == 0:
            print("KOR")
        elif pre == 1:
            print("ENG")
        elif pre == 2:
            print("JP")

        return pre


if __name__ == '__main__':

    model = CnnModel()
    model.train()               # training

    # connect to arduino
    seri = serial.Serial("COM3", 9600)  # (port, speed)
    #seri.open()  # serial port open,

    # use camera
    cam = cv2.VideoCapture(1)  # use first camera
    cam.set(3, 720)  # set width
    cam.set(4, 720)  # set height

    while True:
        while True:
            ret_val, img = cam.read()   # read camera

            if not ret_val:             # if fail to read cam -> break
                break

            cv2.imshow("captured", img)  # show image from camera

            k = cv2.waitKey(60) & 0xFF  # read keyboard push.
            if k == ord('i'):       # when push ' i ' key on keyboard
                break

            if k == ord('x'):       # when push ' x ' key on keyboard
                sys.exit()


        # when push ' i ' key on keyboard
        cv2.imwrite("./Lan.jpg", img)   # read image from cam and save as file(Lan.jpg)

        # read image and image preprocessing
        p_img = Image.open("./Lan.jpg")    # read image file.
        p_img = p_img.convert('L')  # change to black-and-white image.
        p_img = p_img.resize((32, 32))  # resize image to 32 * 32 * 1
        p_data = np.asarray(p_img)  # np type
        p_data = 1 * (p_data > np.mean(p_data))  # change to binary data

        p_data = np.reshape(p_data, [-1, 32, 32, 1])

        # prediction
        p_pre_result = model.predict(p_data)

        print("\n\n", p_pre_result)     # print prediction result.


        # send prediction result to arduino
        seri.write([p_pre_result])  # send data from python to arduino.