import pandas as pd
import numpy as np
import os
import tensorflow
from tensorflow.keras.preprocessing.image import img_to_array, load_img

trainLabels = pd.read_csv("dataset/Training/training_labels.csv")
trainLabels = trainLabels["MEDICINE_NAME"]

testLabels = pd.read_csv("dataset/Testing/testing_labels.csv")
testLabels = testLabels["MEDICINE_NAME"]

valLabels = pd.read_csv("dataset/Validation/validation_labels.csv")
valLabels = valLabels["MEDICINE_NAME"]

trainWords = "dataset/Training/training_words"
testWords = "dataset/Testing/testing_words"
valWords = "dataset/Validation/validation_words"

trainImages = [f for f in os.listdir(trainWords)]
testImages = [f for f in os.listdir(testWords)]
valImages = [f for f in os.listdir(valWords)]

x_train = []
x_test = []
x_val = []

y_train = trainLabels.to_numpy()
y_test = testLabels.to_numpy()
y_val = valLabels.to_numpy()

for trainLabel in os.listdir(trainWords):
    file_path = os.path.join(trainWords, trainLabel)
    image = load_img(file_path, target_size=(64, 64)) ## this is the photo size
    image_array = img_to_array(image, dtype='float32')/255.0 # notmalize between 0 and 1
    x_train.append(image_array)

for testLabel in os.listdir(testWords):
    file_path = os.path.join(testWords, testLabel)
    image = load_img(file_path, target_size=(64, 64)) ## this is the photo size
    image_array = img_to_array(image, dtype='float32')/255.0 # notmalize between 0 and 1
    x_test.append(image_array)

for valLabel in os.listdir(valWords):
    file_path = os.path.join(valWords, valLabel)
    image = load_img(file_path, target_size=(64, 64)) ## this is the photo size
    image_array = img_to_array(image, dtype='float32')/255.0 # notmalize between 0 and 1
    x_val.append(image_array)

x_train = np.array(x_train)
x_test = np.array(x_test)
x_val = np.array(x_val) 

print(y_test)
