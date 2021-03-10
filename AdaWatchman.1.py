from sklearn.ensemble import AdaBoostClassifier 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix 
from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from matplotlib import pyplot
from sklearn import tree 

import numpy as np
import pandas as pd

data1 = pd.read_csv('n9_innerhit_dt_prev_signal.txt', sep=' ', header=None)
data1.columns = ["n9", "inner_hit", "dt_prev_us"]
df_signal = pd.DataFrame(data1)
#this has 495381 rows

data2 = pd.read_csv('n9_innerhit_dt_prev_singles.txt', sep=' ', header=None)
data2.columns = ["n9", "inner_hit", "dt_prev_us"]
df_singles = pd.DataFrame(data2)
#this has 744679 rows 

#df_singles = df_singles.drop(df_singles[df_singles.n9 < 8].index)
#makes a cut and gets rid of data where n9 is less than 8
#df_signal = df_signal.drop(df_signal[df_signal.n9 < 8].index)

#df_singles = df_singles.head(5000)
#df_signal = df_signal.head(5000)
#reduces both files to 5000 items each

#adding labels to data frame where signal = 1 and background = 0
df_signal['label'] = 1
df_singles['label'] = 0

#combining signal and background into one dataframe
all_data = pd.concat([df_signal, df_singles], axis=0)

#sorts data in ascending time order 
#all_data = all_data.sort_values(by=['timestamp'])

#reindexes data to be 0, 1, 2 now order has changed
all_data.index = range(len(all_data))
#all_data.to_csv('all_data_time_order.csv')

#take away time of event from time of previous event
#all_data.timestamp = all_data.timestamp.diff()

#delete first event as this is an error as no previous event to
#takeaway from 
#all_data = all_data.iloc[1:]

#change column name as now time difference 
#all_data = all_data.rename(columns={"timestamp": "t_diff"}) 
#all_data.to_csv('all_data_test.csv')

#now split label column away from rest of data and make an array 
label_column = all_data[['label']]
df_labels = label_column.copy()
array_labels = df_labels.to_numpy()
array_labels = array_labels.flatten()

#now drop label column from rest of data
df_all_data = all_data.drop(['label'], axis=1)
#df_all_data.to_csv('df_all_data_test.csv')

#now have all data organised but seperated into n9 inner hit and t_diff in one 
#file and labels in another file. can not begin algorithm. 

train_X, test_X, train_y, test_y = train_test_split(df_all_data, array_labels, test_size= 0.3, random_state =None)
#data split into test and training sets  

#max_depth = 1 tells us we want our trees to have 1 decision node 
#and 2 leaves 
#n_estimators used to specify number of trees in forest 

classifier = AdaBoostClassifier(
	DecisionTreeClassifier(max_depth=1),
	n_estimators=100,
	learning_rate = 1.0
)

classifier.fit(train_X, train_y)

predictions = classifier.predict(test_X)
#model then used to predict what is signal and what is background 
#given what its learnt 

#finally can evaluate with a confusion matrix

print("confusion matrix is as follows\n", confusion_matrix(test_y, predictions))

print("classification report is as follows\n")
print(classification_report(test_y, predictions)) 

all_probs = classifier.predict_proba(test_X)
#keep only for positive output this tells me signal probabilities 
signal_probs = all_probs[:, 1]

signal_fpr, signal_tpr, thresholds = roc_curve(test_y, signal_probs)

#plot roc curve
pyplot.plot(signal_fpr, signal_tpr, marker='.', label = 'Adaboost')
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.legend()
#pyplot.show()

#pyplot.savefig("roc_curve_timediff_alldata.png")

from sklearn.inspection import permutation_importance

#r = permutation_importance(classifier, test_X, test_y, 
#			n_repeats=30, 
#			random_state=0)

#for i in r.importances_mean.argsort()[::-1]:
#	if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
#		print(f"{df_all_data.columns[i]:<8}"
#			f"{r.importances_mean[i]:.3f}"
#			f" +/- {r.importances_std[i]:.3f}")

#importances = classifier.feature_importances_
#print(importances)

#following code outputs the data with its label column and the label that 
# that the classifier assigned to it 

predicted_label = predictions
test_data = test_X
test_data.loc[:, "classifier"] = predicted_label
#adds the predictions to the test_X dataframe as a column 
print("test data is:")
print(test_data)

rows = test_data.index
print(rows)

print("label column is: ")
print(label_column)

label_column.index = range(len(label_column))
test_labels = label_column.iloc[rows,:]
print(test_labels)

test_data.loc[:, "label"] = test_labels
print(test_data)
test_data.to_csv('classified_dt_prev_with_labels.csv')
