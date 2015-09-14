import csv

import numpy
from sklearn import preprocessing, neighbors, metrics
from sklearn.cross_validation import train_test_split

import knnplots

__author__ = 'bryanhunt'

# Imports for Module 4
# Code common to all modeles from module 3 onwards
# NB. The X and yTransformed variables come from the preprocessing in the previous module.
fileName = "wdbc2.csv"
fileOpen = open(fileName, "rU")
csvData = csv.reader(fileOpen)
dataList = list(csvData)
dataArray = numpy.array(dataList)

X = dataArray[:, 2:32].astype(float)
y = dataArray[:, 1]
# print X.shape
# print y.shape
# print "X dimensions: ", X.shape
# print "Y dimensions: ", y.shape

# print y
le = preprocessing.LabelEncoder()
le.fit(y)
yTransformed = le.transform(y)
# print(yTransformed)

XTrain, XTest, yTrain, yTest = train_test_split(X, yTransformed)

print "XTrain dimensions: ", XTrain.shape
print "yTrain dimensions: ", yTrain.shape
#
print "XTest dimensions: ", XTest.shape
print "yTest dimensions: ", yTest.shape

#
# # print metrics.classi
#
knn = neighbors.KNeighborsClassifier(n_neighbors=3, weights="distance")
knn = knn.fit(XTrain, yTrain)
predicted = knn.predict(XTest)

print metrics.classification_report(yTest, predicted)
print "accuracy:", metrics.accuracy_score(yTest,predicted)

knnplots.plotaccuracy(XTrain,yTrain,XTest,yTest,310)




#
# print "PredictedK3 n_neighbors=3"
# print predictedK3
#
# knnK3 = neighbors.KNeighborsClassifier(n_neighbors=15, weights="distance")
# knnK3 = knnK3.fit(X, yTransformed)
# predictedK15 = knnK3.predict(X)
#
# print "PredictedK15 n_neighbors=15"
# print predictedK15
#
# print "Discrepencies k=3,k=15", predictedK3 != predictedK15
# print "Discrepencies k=3,k=15 count", numpy.sum(predictedK3 != predictedK15)
#
# print "Discrepencies k=3 compared to yTransformed", numpy.sum(predictedK3 != yTransformed)
# print "Discrepencies k=15 compared to yTransformed", numpy.sum(predictedK15 != yTransformed)
#
# print "In this case more neighbours actually performs worse...."
