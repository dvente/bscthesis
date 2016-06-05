import numpy as np
from sklearn import tree
from io import StringIO
# import pydot 


def readData(file):
	vec = []
	label = []
	f = open(file,'r')
	for line in f:
		line = line.rstrip()
		array = line.split(" ")
		if(len(array) == 1): #skip header row
			continue
		vec.append(list(map(float, array[:-1])))
		label.append(int(array[-1]))

	return (vec,label)

weaklearn = tree.DecisionTreeClassifier(max_depth = 1)
X, Y = readData("../generated/train.dat");
X = np.array(X)
Y = np.array(Y)

T = 100
N = len(Y)

# print(X.shape)
# print(Y.shape)

p = np.array([1/N for i in range(N)])
weaklearn.fit(X,Y, sample_weight = p)
# print(weaklearn.predict(X)*Y)

p = np.array([0 for i in range(N)])
p[4] = 1/2
p[5] = 1/2
weaklearn.fit(X,Y, sample_weight = p)
# print(weaklearn.predict(X)*Y)
# print(X[4], Y[4])
# print(X[5], Y[5])


# out = StringIO()
# out = tree.export_graphviz(weaklearn, out_file=out)
# graph = pydot.graph_from_dot_data(dot_data.getvalue())
# graph.write_pdf("iris.pdf")

print(weaklearn.tree_.children_left) #array of left children
print(weaklearn.tree_.children_right) #array of right children
print(weaklearn.tree_.feature) #array of nodes splitting feature
print(weaklearn.tree_.threshold) #array of nodes splitting points
print(weaklearn.tree_.value) #array of nodes values