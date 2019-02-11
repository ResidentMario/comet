from comet_ml import Experiment

experiment = Experiment(api_key="CgFCfEAIYJVIxez3BZzCqFeeX",
                        project_name="fashion_mnist_demo", 
                        workspace="residentmario")

import t4
t4.Package.install("aleksey/fashion-mnist", registry="s3://alpha-quilt-storage", dest=".")

import pandas as pd
data_train = pd.read_csv('aleksey/fashion-mnist/fashion-mnist_train.csv')

# exclude the class label from the training data, otherwise we have nothing to train on
X = data_train.iloc[:, 1:].values

# one-hot encode the classes
y = pd.get_dummies(data_train.iloc[:, 0].values).values

# partition the dataset into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape the flat [0, 255]-entry list into a [0, 1]-entry grid, as desired by the CNN.
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float') / 255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float') / 255

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

clf = Sequential()
clf.add(Conv2D(32, kernel_size=(3, 3),
               activation='relu',
               kernel_initializer='he_normal',
               input_shape=(28, 28, 1)))
clf.add(MaxPooling2D((2, 2)))
clf.add(Dropout(0.25))
clf.add(Conv2D(64, (3, 3), activation='relu'))
clf.add(MaxPooling2D(pool_size=(2, 2)))
clf.add(Dropout(0.25))
clf.add(Conv2D(128, (3, 3), activation='relu'))
clf.add(Dropout(0.4))
clf.add(Flatten())
clf.add(Dense(128, activation='relu'))
clf.add(Dropout(0.3))
clf.add(Dense(10, activation='softmax'))

clf.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adam(),
            metrics=['accuracy'])

history = clf.fit(X_train, y_train, batch_size=512, epochs=1, verbose=1, validation_data=(X_test, y_test))

y_test_pred = clf.predict_classes(X_test)

# import numpy as np
# y_test_classed = np.nonzero(y_test)[1]
# from sklearn.metrics import classification_report
# print(classification_report(y_test_pred, y_test_classed, target_names=img_class_key.values()))

clf.save('clf.h5')