import csv

import matplotlib.pyplot as plt
import numpy
import scipy
from sklearn import preprocessing

__author__ = 'bryanhunt'

# Imports for Module 4
# Code common to all modeles from module 3 onwards
# NB. The X and yTransformed variables come from the preprocessing in the previous module.
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

# c uses the number of classes and automatically assigns color based upon
plt.scatter(x = X[:,0], y = X[:,1],c=y)
plt.xlabel("radius")
plt.xlabel("texture")
plt.show()


