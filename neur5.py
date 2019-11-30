from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Input, Conv2D, MaxPooling2D, Flatten
from keras.utils import np_utils
from keras.regularizers import l2
dataset_train= np.loadtxt("neur2.csv", delimiter=",")
dataset_test= np.loadtxt("neur_test.csv", delimiter=",")

(Y_train, X_train)= (dataset_train[:,0],dataset_train[:,1:28])
(Y_test, X_test)= (dataset_test[:,0],dataset_test[:,1:28])
#print("Original Xte shape", X_test.shape)
#print("Original Yte shape", Y_test.shape)
print("Original X shape", X_train.shape)
print("Original Y shape", Y_train.shape)
print("Original X shape", X_test.shape)
print("Original Y shape", Y_test.shape)




X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
Y_train = Y_train.astype('float32')
Y_test = Y_test.astype('float32')

print("Training X matrix shape", X_train.shape)
print("Testing X matrix shape", X_test.shape)

# Represent the targets as one-hot vectors: e.g. 2 -> [0, 0, 1, 0, 0, 0, 0, 0, 0].
nb_classes = 2 # (for 2 classes, better to just have a sigmoidal output) (in tensorflow for the 2 class one, just use the sigmoid function cause the softmax function goes to infinity)
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test  = np_utils.to_categorical(Y_test, nb_classes)
print("Training Y matrix shape", Y_train.shape)
print("Testing Y matrix shape", Y_test.shape)

model = Sequential()
model.add(Dense(10000, activation='sigmoid', input_shape=(27,))) # Use input_shape=(28,28) for unflattened data. #kernel_regularizer=l2(0.001)
#model.add(Dropout(0.2)) # Including dropout layer helps avoid overfitting.
model.add(Dense(6, activation='sigmoid')) # Use softmax layer for multi-class problems.
model.add(Dense(2, activation='softmax')) # Use softmax layer for multi-class problems.
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# dashboard
import os
import time
from keras.callbacks import TensorBoard

# tensorboard --logdir=nova:C:\Users\petrb\nova\nova-keras-tutorials\tb_log

log_dir = './tb_log/' + time.strftime("%c")
log_dir = log_dir.replace(' ', '_').replace(':', '-')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
tb = TensorBoard(log_dir=log_dir,
                 histogram_freq=1,
                 write_graph=True,
                 write_grads=False,
                 write_images=False)
history = model.fit(X_train, Y_train, batch_size=128, epochs=10, verbose=1,
                    validation_data=(X_test, Y_test), callbacks=[tb])

# Note: when calling evaluate, dropout is automatically turned off.
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test cross-entropy loss: %0.5f' % score[0])
print('Test accuracy: %0.2f' % score[1])

# Plot loss trajectory throughout training.
plt.figure(1, figsize=(14,5))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='valid')
plt.xlabel('Epoch')
plt.ylabel('Cross-Entropy Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['acc'], label='train')
plt.plot(history.history['val_acc'], label='valid')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()