import csv

import matplotlib.pyplot as plt

import numpy
from sklearn import preprocessing

# making a heatmap of the breast cancer data to determine colinearity
__author__ = 'bryanhunt'

fileName = "wdbc.csv"
fileOpen = open(fileName, "rU")
csvData = csv.reader(fileOpen)
dataList = list(csvData)
dataArray = numpy.array(dataList)

X = dataArray[:, 2:32].astype(float)
y = dataArray[:, 1]
print X.shape
print y.shape
print "X dimensions: ", X.shape
print "Y dimensions: ", y.shape

print y
le = preprocessing.LabelEncoder()
le.fit(y)
yTransformed = le.transform(y)
print(yTransformed)

correlationMatrix = numpy.corrcoef(X, rowvar=0)

fig, ax = plt.subplots()

heatmap = ax.pcolor(correlationMatrix, cmap=plt.cm.Blues)
plt.colorbar(heatmap)
plt.show()
