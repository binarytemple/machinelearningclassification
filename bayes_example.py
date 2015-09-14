import csv

import numpy
from sklearn import preprocessing, metrics
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB

__author__ = 'bryanhunt'

fileName = "wdbc.csv"
fileOpen = open(fileName, "rU")
csvData = csv.reader(fileOpen)
dataList = list(csvData)
dataArray = numpy.array(dataList)

X = dataArray[:, 2:32].astype(float)
y = dataArray[:, 1]

le = preprocessing.LabelEncoder()
le.fit(y)
yTransformed = le.transform(y)
# print(yTransformed)

XTrain, XTest, yTrain, yTest = train_test_split(X, yTransformed, random_state=1)

print "XTrain dimensions: ", XTrain.shape
print "yTrain dimensions: ", yTrain.shape
#
print "XTest dimensions: ", XTest.shape
print "yTest dimensions: ", yTest.shape

nbmodel = GaussianNB().fit(XTrain, yTrain)
predicted = nbmodel.predict(XTest)

mat = metrics.confusion_matrix(yTest, predicted)
print mat