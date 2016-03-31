#!/usr/bin/python3
import numpy as np
import sys
from sklearn.tree import DecisionTreeClassifier

#python AdaBoost.py <max nodes> <number of test cases>

def isDist(array):
	return (sum(list(map(lambda x: x<0,array)))==0 and np.isclose(sum(array),1.0))

def uniformDist(N):
	w = [float(1/N) for i in range(0,N)]
	return w

def readData(file):
	vec = []
	label = []
	f = open(file,'r')
	for line in f:
		line = line.rstrip()
		array = line.split(" ")
		if(len(array) == 1):
			continue
		vec.append(list(map(float, array[:-1])))
		label.append(int(array[-1]))

	return (vec,label)

def conv(inp):
	if inp:
		return 1
	else:
		return 0

test = int(sys.argv[2])
boostErr = 0
vec, label = readData("../generated/train.dat")
vec = np.array(vec)
label = np.array(label).reshape(-1,1)


if(sys.argv[1]=="None"):
	a = DecisionTreeClassifier()
else:
	a = DecisionTreeClassifier(max_depth = max(np.log2(int(sys.argv[1])),1))

a.fit(vec,label)

for t in range(0,test):
	inp = np.array(np.random.normal(0,1,10)).reshape(1,-1)
	#print(list(map(lambda x: x**2, vec)))
	true = conv((np.sum(list(map(lambda x: x**2, inp)))) > 9.34)
	if a.predict(inp) != true:
		boostErr += 1

print( a.tree_.node_count, boostErr/float(test)) 	

