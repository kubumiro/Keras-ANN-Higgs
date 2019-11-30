#tensorboard --logdir=C:\Users\User\PycharmProjects\neur\tb_log

import plotly
import plotly.graph_objs as go
from plotly.offline import plot
from matplotlib import pyplot as plt
import numpy as np

from sklearn.metrics import roc_auc_score, roc_curve, auc
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2
import pandas as pd


df = pd.read_csv('higgs.csv', nrows=15000)




y = df.iloc[:, 0]
X = df.iloc[:, 1:]



from sklearn import preprocessing
X = preprocessing.normalize(X, norm='l2')


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.30, random_state=42)

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

from keras import regularizers, optimizers
from keras import regularizers
model = Sequential()
model.add(Dense(2500, activation='tanh', input_shape=(28,), kernel_regularizer=regularizers.l2(0.001),activity_regularizer=regularizers.l1(0.001)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

#sgd = optimizers.SGD(lr=0.01, decay=0.0000001, momentum=0.9, nesterov=True)
model.compile(loss='MSE', optimizer='adam', metrics=['accuracy'])
model.summary()

import os
import time
from keras.callbacks import TensorBoard

log_dir = './tb_log/' + time.strftime("%c")
log_dir = log_dir.replace(' ', '_').replace(':', '-')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
tb = TensorBoard(log_dir=log_dir,
                 histogram_freq=1,
                 write_graph=True,
                 write_grads=False,
                 write_images=False)
history = model.fit(X_train, Y_train, batch_size=20, epochs=20, verbose=1,
                    validation_data=(X_test, Y_test), callbacks=[tb])



score = model.evaluate(X_test, Y_test, verbose=0)
print('Test MSE loss: %0.5f' % score[0])
print('Test accuracy: %0.2f' % score[1])



plt.figure(1, figsize=(14,5))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='valid')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['acc'], label='train')
plt.plot(history.history['val_acc'], label='valid')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

Y_pred=model.predict(X_test)
#print(Y_pred.type()) numpy ndarray
yyy=roc_auc_score(Y_test, Y_pred)
xxx=roc_curve(Y_test, Y_pred)
#print(yyy)
#print(xxx)
#fpr["micro"], tpr["micro"], _=roc_curve(Y_test, Y_pred)
fpr = dict()
tpr = dict()
roc_auc = dict()
#print(Y_pred)
from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(Y_test, Y_pred)
#print(Y_pred.shape)
# Compute micro-average ROC curve and ROC area
#fpr["micro"], tpr["micro"], _ = roc_curve(Y_test, Y_pred)
#roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


#print(fpr[2])
#print(tpr[2])
plt.plot(fpr,tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(label=yyy)
plt.show()

#print((X_train[Y_train>0.5]))


Classifier_training_S = model.predict(X_train[Y_train>0.5])
Classifier_training_B = model.predict(X_train[Y_train<0.5])
Classifier_testing_S = model.predict(X_test[Y_test>0.5])
Classifier_testing_B = model.predict(X_test[Y_test<0.5])

c_max = 1
c_min = 0

# Get histograms of the classifiers
Histo_training_S = np.histogram(Classifier_training_S, bins=40, range=(c_min, c_max))
Histo_training_B = np.histogram(Classifier_training_B, bins=40, range=(c_min, c_max))
Histo_testing_S = np.histogram(Classifier_testing_S,bins=40,range=(c_min,c_max))
Histo_testing_B = np.histogram(Classifier_testing_B,bins=40,range=(c_min,c_max))

# Lets get the min/max of the Histograms
AllHistos = [Histo_training_S, Histo_training_B, Histo_testing_S, Histo_testing_B]
h_max = max([histo[0].max() for histo in AllHistos]) * 1.2
h_min = max([histo[0].min() for histo in AllHistos])

# Get the histogram properties (binning, widths, centers)
bin_edges = Histo_training_S[1]
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.
bin_widths = (bin_edges[1:] - bin_edges[:-1])

# To make error bar plots for the data, take the Poisson uncertainty sqrt(N)
ErrorBar_testing_S = np.sqrt(Histo_testing_S[0])
ErrorBar_testing_B = np.sqrt(Histo_testing_B[0])

# Draw objects
ax1 = plt.subplot(111)

# Draw solid histograms for the training data
ax1.bar(bin_centers - bin_widths / 2., Histo_training_S[0], facecolor='blue', linewidth=0, width=bin_widths,
        label='S (Train)', alpha=0.5)
ax1.bar(bin_centers - bin_widths / 2., Histo_training_B[0], facecolor='red', linewidth=0, width=bin_widths,
        label='B (Train)', alpha=0.5)

# # Draw error-bar histograms for the testing data
ax1.errorbar(bin_centers, Histo_testing_S[0], yerr=ErrorBar_testing_S, xerr=None, ecolor='blue', c='blue', fmt='o',
             label='S (Test)')
ax1.errorbar(bin_centers, Histo_testing_B[0], yerr=ErrorBar_testing_B, xerr=None, ecolor='red', c='red', fmt='o',
             label='B (Test)')

# Make a colorful backdrop to show the clasification regions in red and blue
ax1.axvspan(0.0, c_max, color='blue', alpha=0.08)
ax1.axvspan(c_min, 0.0, color='red', alpha=0.08)

# Adjust the axis boundaries (just cosmetic)
ax1.axis([c_min, c_max, h_min, h_max])

# Make labels and title
plt.title("Classification with scikit-learn")
plt.xlabel("Classifier, SVM [rbf kernel, C=1, gamma=0.005]")
plt.ylabel("Counts/Bin")

# Make legend with smalll font
legend = ax1.legend(loc='upper center', shadow=True, ncol=2)
for alabel in legend.get_texts():
    alabel.set_fontsize('small')

plt.show()

print('AUC=  ')
print(yyy)



df = pd.read_csv('higgs.csv', usecols=[0], nrows=25000)


df.rename(columns={0: 'target'})

df.rename(columns={0: 'target'}, inplace=True)
df.head()

v=(df.sum())/25000
print('Signal-Background ratio=  ')
print(v)

'''
trace0 = go.Scatter(
    #x = history.history['acc'],
    y = history.history['acc'],
    mode = 'lines',
    name = 'lines'
)
fig = dict(data=trace0)
py.iplot(fig, filename='simple-connectgaps')
from keras.utils import plot_model
plot_model(model, show_shapes=True, to_file='filename.jpg')'''