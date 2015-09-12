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

yFreq = scipy.stats.itemfreq(y)
print yFreq
plt.bar(left=0, height=int(yFreq[0][1]))
plt.bar(left=1, height=int(yFreq[1][1]))
plt.xlabel("diagnosis")
plt.ylabel("frequency")
plt.show()


