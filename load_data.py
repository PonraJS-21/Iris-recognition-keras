
import pickle
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import os


label = os.listdir("normalized_dataset_image")
dataset = []
for image_label in label:

    images = os.listdir("normalized_dataset_image/"+image_label)

    for image in images:
        img = cv2.imread("normalized_dataset_image/"+image_label+"/"+image)
        img = cv2.resize(img, (80, 360))
        dataset.append((img, image_label))


X = []
Y = []

for input, image_label in dataset:

    X.append(input)

    Y.append(label.index(image_label))


X = np.array(X)
Y = np.array(Y)


X_train, y_train,  = X, Y


data_set = (X_train, y_train)


save_label = open("int_to_word_out.pickle", "wb")
pickle.dump(label, save_label)
save_label.close()
