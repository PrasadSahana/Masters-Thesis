import pandas as pd
import numpy as np
import tensorflow as tf
from keras import models, layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

dataframe1 = pd.read_excel("/Users/sahanaprasad/Desktop/Thesis/Thesis/data/2_classes/Object_Testing.xlsx", header= None)
data1 = np.array(dataframe1)


dataframe2 = pd.read_excel("/Users/sahanaprasad/Desktop/Thesis/Thesis/data/2_classes/Object_Testing.xlsx", header= None)
data2 = np.array(dataframe2)


normdata1 = np.zeros((data1.shape[0],data1.shape[1]))
for i in range(data1.shape[0]):
    normdata1[i] = data1[i]/max(abs(data1[i]))

normdata2 = np.zeros((data2.shape[0],data2.shape[1]))
for i in range(data2.shape[0]):
    normdata2[i] = data2[i]/max(abs(data2[i]))

num_classes = 2

X = np.vstack((normdata1,normdata2)) # Features
Y = np.hstack((np.zeros(normdata1.shape[0]),np.ones(normdata2.shape[0]))) # Labels

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.4, random_state = 36)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

model = models.Sequential()
model.add(layers.Input(shape= x_train.shape[1:]))
model.add(layers.Conv1D(128, 7, activation='relu', padding='same'))
model.add(layers.Dropout(0.2))
model.add(layers.MaxPooling1D(pool_size= 6,padding='same'))
model.add(layers.Conv1D(64, 3, activation='relu', padding='same'))
model.add(layers.Dropout(0.2))
model.add(layers.MaxPooling1D(pool_size= 6, padding='same'))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.RMSprop(0.001), metrics=['accuracy'])

model.load_weights('/Users/sahanaprasad/Desktop/Thesis/Thesis_spyder/cnn1d.h5')
model.summary()

# Train the model
print('Training the model:')
history = model.fit(x_train, y_train, epochs= 30, validation_split= 0.1, verbose= 1)

# Test the model after training
print('Testing the model:')
model.evaluate(x_test, y_test, verbose= 1)

#Plot Accuracy
plt.plot(history.history['accuracy'], color='blue')
plt.plot(history.history['val_accuracy'], color='black')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#Plot Loss
plt.plot(history.history['loss'], color='blue')
plt.plot(history.history['val_loss'], color='black')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Confusion Matrix
labels = ["Nailbed#1-4", "Nailbed#5"]
y_actual = y_test
y_est = model.predict(x_test)
y_est = np.argmax(y_est, axis= 1)
ConfusionMatrix = confusion_matrix(y_actual, y_est).ravel()
TrueNeg, FalsePos, FalseNeg, TruePos  = ConfusionMatrix
disp = ConfusionMatrixDisplay(confusion_matrix=ConfusionMatrix.reshape(2,2), display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion matrix for 30cm 0°')


TruePosRate = TruePos/(TruePos + FalseNeg)
TrueNegRate = TrueNeg/(TrueNeg + FalsePos)
FalseDiscRate = FalsePos/(FalsePos + TruePos)
NegPredVal = TrueNeg/(TrueNeg + FalseNeg)
accuracy = round((TruePos+TrueNeg)/(TruePos+TrueNeg+FalsePos+FalseNeg)*100)
precision = round((TruePos/(TruePos+FalsePos))*100)
recall = round((TruePos/(TruePos+FalseNeg))*100)
F1_Score = round(2 * (precision * recall)/(recall + precision))


print("The accuracy of CNN model is : " + str(accuracy) + "%")
print("The precision of CNN model is : " + str(precision) + "%")
print("The recall of CNN model is : " + str(recall) + "%")
print("The F1_Score of CNN model is : " + str(F1_Score) + "%")

# creating the bar plot
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
langs = ['TPR', 'TNR', 'FDR', 'NPV']

TPR = round(TruePosRate*100)
TNR = round(TrueNegRate*100)
FDR = round(FalseDiscRate*100)
NPV = round(NegPredVal*100)
students = [TPR, TNR, FDR, NPV]
ax.bar(langs,students)

for i in range(len(students)):
    plt.annotate(str(students[i])+'%', xy=(langs[i],students[i]), ha='center', va='bottom')
    
plt.show()


