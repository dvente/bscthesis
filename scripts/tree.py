#!/usr/bin/python3
import numpy as np
import sys
from sklearn.tree import DecisionTreeClassifier
import argparse

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


parser = argparse.ArgumentParser(description='AdaBoost trainer')
parser.add_argument('-n', '--nodes', type = int, default = 0, help='Maximum number of nodes of the tree, use 0 for unlimited nodes.')
parser.add_argument('--trainData', default = "../generated/train.dat",help="location of the training data")
parser.add_argument("--testData", default = "../generated/test.dat", help = "location of the test data" )
parser.add_argument("-v", "--verbose", help="display number of traing and test cases aswell", action="store_true")

args = parser.parse_args()

boostErr = 0
vec, label = readData(args.trainData)
vec = np.array(vec)
label = np.array(label).reshape(-1,1)


if(args.nodes == 0):
	a = DecisionTreeClassifier()
else:
	a = DecisionTreeClassifier(max_depth = max(np.log2(args.nodes),1))

a.fit(vec,label)

testVec, testLabel = readData(args.testData);
testVec = np.array(testVec)
testLabel = np.array(testLabel)
for t in range(0,len(testVec)):
	if(a.predict(testVec[t].reshape(1,-1)) != testLabel[t]):
		boostErr += 1

if(args.verbose):
	print(a.N, len(testVec), a.tree_.node_count, boostErr/len(testVec) )
else:
	print(a.tree_.node_count, boostErr/len(testVec)) 	


