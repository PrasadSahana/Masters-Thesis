import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,f1_score, ConfusionMatrixDisplay
from sklearn.metrics import plot_confusion_matrix,roc_curve,auc
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import pickle
from itertools import cycle



object1_data = pd.DataFrame()
object2_data = pd.DataFrame() 
object3_data = pd.DataFrame()
object4_data = pd.DataFrame()
object5_data = pd.DataFrame()
training_set = pd.DataFrame()
training_set1 = pd.DataFrame() 
training_set2 = pd.DataFrame()  
training_set3 = pd.DataFrame() 
training_set4 = pd.DataFrame()
training_set5 = pd.DataFrame() 
validation_set = pd.DataFrame()
validation_set1 = pd.DataFrame() 
validation_set2 = pd.DataFrame() 
validation_set3 = pd.DataFrame() 
validation_set4 = pd.DataFrame() 
validation_set5 = pd.DataFrame() 


dataframe1 = pd.read_excel("/Users/sahanaprasad/Desktop/Thesis/Thesis/data/5_classes/Object1_Training.xlsx", header= None).dropna()
#dataframe1 = pd.read_excel("/Users/sahanaprasad/Desktop/Thesis/Thesis/data/Nailbed1&Nailbed5_AfterAC/30cm0deg/5_classes/Object1_Training.xlsx", header= None).dropna()#This is for AutoCorrelation function
#dataframe1 = pd.read_excel("/Users/sahanaprasad/Desktop/Thesis/Thesis/data/Combined_Nailbeds/3_classes/Object1_30cm_0deg_ALL_Training.xlsx", header= None).dropna()#This is for combined

dataframe1.insert(0, "Object Type", "NB#1")
object1_data = object1_data.append(dataframe1,ignore_index=True)
training_set1, validation_set1 = train_test_split(object1_data, test_size=0.3, random_state=42)

dataframe2 = pd.read_excel("/Users/sahanaprasad/Desktop/Thesis/Thesis/data/5_classes/Object2_Training.xlsx", header= None)
#dataframe2 = pd.read_excel("/Users/sahanaprasad/Desktop/Thesis/Thesis/data/Nailbed1&Nailbed5_AfterAC/30cm0deg/5_classes/Object2_Training.xlsx", header= None)#This is for AutoCorrelation function
#dataframe2 = pd.read_excel("/Users/sahanaprasad/Desktop/Thesis/Thesis/data/Combined_Nailbeds/3_classes/Object1_30cm_-8deg_ALL_Training.xlsx", header= None).dropna()#This is for combined

dataframe2.insert(0, "Object Type", "NB#2")
object2_data = object2_data.append(dataframe2,ignore_index=True)
training_set2, validation_set2 = train_test_split(object2_data, test_size=0.3, random_state=42)

dataframe3 = pd.read_excel("/Users/sahanaprasad/Desktop/Thesis/Thesis/data/5_classes/Object3_Training.xlsx", header= None)
#dataframe3 = pd.read_excel("/Users/sahanaprasad/Desktop/Thesis/Thesis/data/Nailbed1&Nailbed5_AfterAC/30cm0deg/5_classes/Object3_Training.xlsx", header= None)#This is for AutoCorrelation function
#dataframe3 = pd.read_excel("/Users/sahanaprasad/Desktop/Thesis/Thesis/data/Combined_Nailbeds/3_classes/Object1_30cm_10deg_ALL_Training.xlsx", header= None).dropna()#This is for combined

dataframe3.insert(0, "Object Type", "NB#3")
object3_data = object3_data.append(dataframe3,ignore_index=True)
training_set3, validation_set3 = train_test_split(object3_data, test_size=0.3, random_state=42)


dataframe4 = pd.read_excel("/Users/sahanaprasad/Desktop/Thesis/Thesis/data/5_classes/Object4_Training.xlsx", header= None)
#dataframe4 = pd.read_excel("/Users/sahanaprasad/Desktop/Thesis/Thesis/data/Nailbed1&Nailbed5_AfterAC/30cm0deg/5_classes/Object4_Training.xlsx", header= None)#This is for AutoCorrelation function
#dataframe4 = pd.read_excel("/Users/sahanaprasad/Desktop/Thesis/Thesis/data/Combined_Nailbeds/3_classes/Object1_40cm_5deg_ALL_Training.xlsx", header= None).dropna()#This is for combined

dataframe4.insert(0, "Object Type", "NB#4")
object4_data = object4_data.append(dataframe4,ignore_index=True)
training_set4, validation_set4 = train_test_split(object4_data, test_size=0.3, random_state=42)

dataframe5 = pd.read_excel("/Users/sahanaprasad/Desktop/Thesis/Thesis/data/5_classes/Object5_Training.xlsx", header= None)
#dataframe5 = pd.read_excel("/Users/sahanaprasad/Desktop/Thesis/Thesis/data/Nailbed1&Nailbed5_AfterAC/30cm0deg/5_classes/Object5_Training.xlsx", header= None)#This is for AutoCorrelation function
#dataframe5 = pd.read_excel("/Users/sahanaprasad/Desktop/Thesis/Thesis/data/Combined_Nailbeds/3_classes/Object1_40cm_8deg_ALL_Training.xlsx", header= None).dropna()#This is for combined

dataframe5.insert(0, "Object Type", "NB#5")
object5_data = object5_data.append(dataframe5,ignore_index=True)
training_set5, validation_set5 = train_test_split(object5_data, test_size=0.3, random_state=42)


#Appending training data from all the object types        
training_set = training_set.append(training_set1,ignore_index=True)
training_set = training_set.append(training_set2,ignore_index=True)
training_set = training_set.append(training_set3,ignore_index=True)
training_set = training_set.append(training_set4,ignore_index=True)
training_set = training_set.append(training_set5,ignore_index=True)

#Appending validation data from all the object types  
validation_set = validation_set.append(validation_set1,ignore_index=True)
validation_set = validation_set.append(validation_set2,ignore_index=True)
validation_set = validation_set.append(validation_set3,ignore_index=True)
validation_set = validation_set.append(validation_set4,ignore_index=True)
validation_set = validation_set.append(validation_set5,ignore_index=True)

X_train = training_set.iloc[:, 1:].values
Y_train = training_set.iloc[:, 0].values
X_val = validation_set.iloc[:, 1:].values
y_val = validation_set.iloc[:, 0].values

#Initializing the MLPClassifier
classifier = MLPClassifier(hidden_layer_sizes=(100,200,600,600,600,600,600), max_iter=600, activation='relu', solver='adam',random_state=46)

#Fitting the training data to the network
classifier.fit(X_train, Y_train)

#Predicting y for X_val
y_pred = classifier.predict(X_val)

#Comparing the predictions against the actual observations in y_val
#cm = confusion_matrix(y_pred, y_val)
#print(cm, "\n")

#Ploting confusion matrix
#disp = plot_confusion_matrix(classifier, X_val, y_val)
#disp.plot(cmap=plt.cm.OrRd)
#plt.title('Confusion matrix for all Nailbeds at 30cm 0°')

#Printing the accuracy
#accuracy = accuracy_score(y_pred, y_val)
#accuracy_percent = round(accuracy * 100)
#print("Accuracy of MLP Classifier : ", accuracy_percent, "\n")

#Printing F1
print(classification_report(y_pred,y_val) , "\n")

#FP = cm.sum(axis=0) - np.diag(cm) 
#FN = cm.sum(axis=1) - np.diag(cm)
#TP = np.diag(cm)
#TN = cm.sum() - (FP + FN + TP)
#FP = FP.astype(float)
#FN = FN.astype(float)
#TP = TP.astype(float)
#TN = TN.astype(float)

#print("TruePostive(TP) : ", TP, "\n")
#print("TrueNegative(TN) : ", TN, "\n")
#print("FalsePostive(FP) : ", FP, "\n")
#print("False Negative(FN) : ", FN, "\n")



# Sensitivity, hit rate, recall, or true positive rate
#TPR = TP/(TP+FN)
# Specificity, selectivity or true negative rate
#TNR = TN/(TN+FP)
# False discovery rate
#FDR = FP/(TP+FP)
# Negative predictive value
#NPV = TN/(TN+FN)

#fpr = dict()
#tpr = dict()
#roc_auc = dict()
#n_classes = 5

#for i in range(n_classes):
    #fpr[i], tpr[i], _ = roc_curve(np.array(pd.get_dummies(y_val))[:, i], np.array(pd.get_dummies(y_pred))[:, i])
    #roc_auc[i] = auc(fpr[i], tpr[i])

#all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

#mean_tpr = np.zeros_like(all_fpr)
#for i in range(n_classes):
    #mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

#mean_tpr /= n_classes

#fpr["macro"] = all_fpr
#tpr["macro"] = mean_tpr
#roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

#lw=2
#plt.figure(figsize=(8,5))
#plt.plot(fpr["macro"], tpr["macro"],
#label='macro-average ROC curve (area = {0:0.2f})'
#''.format(roc_auc["macro"]),
#color='green', linestyle='dotted', linewidth=4)
#colors = cycle(['purple', 'pink', 'blue', 'red', 'yellow'])
#for i, color in zip(range(n_classes), colors):
    #plt.plot(fpr[i], tpr[i], color=color, lw=lw, 
             #label='ROC curve of Nailbed #{0} (area = {1:0.2f})'
             #''.format(i + 1 ,  round(roc_auc[i])))
    #print("roc_auc_score of Nailbed #" , i+1, ": ", roc_auc[i])

#print("\n")
#plt.plot([0, 1], [0, 1], 'k--', lw=lw)
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.annotate('Random Guess',(.5,.48))
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('ROC Curve')
#plt.legend(loc="lower right")



#Saving the model in a pickle file
filename = '/Users/sahanaprasad/Desktop/Thesis/Thesis/GUI_Testing_data/MLP/mlp_model.pkl'
pickle.dump(classifier, open(filename, 'wb'))

