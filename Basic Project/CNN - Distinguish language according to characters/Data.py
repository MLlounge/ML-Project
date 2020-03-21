# you can watch more on ' www.mllounge.com '
# make training and test data

from PIL import Image
import os, glob
import numpy as np
from sklearn.model_selection import train_test_split

dir = "./Language_Data"     # place of image directory
cat = ["KOR", "ENG", "JP"] # category korean, english, japanese
nb_classes = len(cat)   # number of category
img_w = 32      # width of image
img_h = 32      # height og image
pixels = img_w * img_h * 1  # size of image.
x = []          # array to save image data
y = []          # array to save label


# 1. make parameter cat's each value as one-hot's label
# 2. read image from each category folder
# 3. and then make input data 'x' and match output label 'y'
for idx, cate in enumerate(cat):
    label = [0 for i in range(nb_classes)]  # make one-hot label. initialize ont-hot label.
                                            # [0, 0, 0]
    label[idx] = 1                          # set one-hot by category
                                            # kor = [1, 0, 0], eng = [0, 1, 0], jp = [0, 0, 1]
    img_dir = dir + "/" + cate              # set each language folder name
    files = glob.glob(img_dir + "/*.jpg")   # read each language folder's files(images)


    for i, f in enumerate(files):           # make input data 'x' and match output label 'y'
        img = Image.open(f)                 # read each image file.
        img = img.convert('L')              # change image to black-and-white image
        img = img.resize((img_w, img_h))    # resize image size to 32 * 32 * 1

        for ang in range(-20, 21, 5):   # image argumentation. # use rotated image for learning.
                                        # add little bit changed image for more exact learning.
            img2 = img.rotate(ang)      # rotate img [-20, -15, -10, ... , 15, 20] degrees
            data = np.asarray(img2)              # change image to np type.

            data = 1 * (data > np.mean(data))   # change to binary data

            x.append(data)                      # append image data to array x.
            y.append(label)                     # append label to array y.

x = np.array(x)
y = np.array(y)

x_train, x_test, y_train, y_test = train_test_split(x, y)   # distribute train and test data.
xy = (x_train, x_test, y_train, y_test)

np.save("./image/data.npy", xy)     # save image and label data as file. file name = data.npy