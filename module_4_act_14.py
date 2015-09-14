import csv

import numpy
from sklearn import neighbors

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
print X.shape
print y.shape
print "X dimensions: ", X.shape
print "Y dimensions: ", y.shape

nbrs = neighbors.NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(X)
distances, indices = nbrs.kneighbors(X)

print 'neighbors  \n- %s \n distances \n %s ' % (indices[:5],distances[:5] )

