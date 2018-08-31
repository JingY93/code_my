import pandas as pd
import numpy as py
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
dataset = pd.read_csv('studentscores.csv')
X = dataset.iloc[ : ,   : 1 ].values
Y = dataset.iloc[ : , 1 ].values
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 1/4, random_state = 0)
# dataset = pd.read_csv('studentscores.csv')
# X = dataset.iloc[:,:1].values
# Y = dataset.iloc[:,1].values
model = Sequential()
model.add(Dense(1, input_dim=1))
model.compile(optimizer='sgd',loss='mse')

print('Training -----------')

model.fit(X_train, Y_train, epochs=300, batch_size=10)

print('\nTesting ------------')

cost = model.evaluate(X_test, Y_test)

print('test cost:', cost)
# print('test accuracy: ', accuracy)

W, b = model.layers[0].get_weights()

print('Weights=', W, '\nbiases=', b)

Y_pred = model.predict(X_test)

plt.scatter(X_test, Y_test)

plt.plot(X_test, Y_pred)

plt.show()
