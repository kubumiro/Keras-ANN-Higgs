import pandas as pds
import matplotlib.pyplot as plt
import numpy as np

from keras.layers import Dense, Dropout, Activation, Input, Conv2D, MaxPooling2D, Flatten


dataframeXv = (pds.read_csv('neur2.csv', usecols=[1]))
dataframeYv = (pds.read_csv('neur2.csv', usecols=[4]))
dataframeZ = (pds.read_csv('neur2.csv', usecols=[0]))

dataframeX = np.array(pds.read_csv('neur2.csv', usecols=[1]))
dataframeY = np.array(pds.read_csv('neur2.csv', usecols=[4]))
dataframeZ = np.array(pds.read_csv('neur2.csv', usecols=[0]))

nb_classes = 2 # (for 2 classes, better to just have a sigmoidal output) (in tensorflow for the 2 class one, just use the sigmoid function cause the softmax function goes to infinity)
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test  = np_utils.to_categorical(Y_test, nb_classes)


from keras.models import Sequential
from keras.layers import Dense
model = Sequential()
model.add(Dense(12, input_shape=(1,), kernel_initializer="uniform", activation='sigmoid'))
model.add(Dense(12, kernel_initializer="uniform", activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(12, kernel_initializer="uniform", activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(2, kernel_initializer="uniform", activation='sigmoid'))
model.summary()

# 3
import keras
tbCallBack = keras.callbacks.TensorBoard(log_dir='tmp/keras_logs', write_graph=True)

# 4
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(dataframeXv.values, dataframeYv.values, epochs=10, batch_size=50,  verbose=1, validation_split=0.3, callbacks=[tbCallBack])



for i in range (0,300):
    if dataframeZ[i]==1:
        plt.plot(dataframeX[i],dataframeY[i],'ro')
    else:
        plt.plot(dataframeX[i], dataframeY[i], 'bs')

#plot_decision_boundary(np.column_stack((dataframeX,dataframeY)), dataframeZ, model, steps=1000, cmap='Paired')
plt.show()


