__author__ = 'bryanhunt'


import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data
labels = iris.target_names
#Symbols to represent the points for the three classes on the graph.
gMarkers = ["+", "_", "x"]
#Colours to represent the points for the three classes on the graph
gColours = ["blue", "magenta", "cyan"]
#The index of the class in target_names
gIndices = [0, 1, 2]

counter=0



for x in [0,1,2,3]:
    for y in [0,1,2,3]:
        #Column indices for the two features you want to plot against each other:
        f1 = x
        f2 = y
        plt.subplot(4,4,counter)
        counter = counter + 1
        for mark, col, i, iris.target_name in zip(gMarkers, gColours, gIndices, labels):
            plt.scatter(x = X[iris.target == i, f1], y = X[iris.target == i, f2], marker = mark, c = col, )

        plt.legend(loc='upper right')
        plt.xlabel(iris.feature_names[f1],{'fontsize': '8'})
        plt.ylabel(iris.feature_names[f2],{'fontsize': '8'})


# plt.plot()
plt.tight_layout()
plt.show()

