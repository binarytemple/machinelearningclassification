import csv
import numpy
from sklearn import preprocessing

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
