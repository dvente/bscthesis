#!/usr/bin/python3.4
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import argparse

def isDist(array):
	return (array>=0 and sum(array)==1)

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

parser = argparse.ArgumentParser(description='AdaBoost trainer using the Scikit-learn implementation')
parser.add_argument('trails', metavar = 'T', type = int, help='Number of trails to train AdaBoost with')
parser.add_argument('--trainData', default = "../generated/train.dat",help="location of the training data")
parser.add_argument("--testData", default = "../generated/test.dat", help = "location of the test data" )
parser.add_argument("-v", "--verbose", help="display number of traing and test cases aswell", action="store_true")

args = parser.parse_args()

boostErr = 0

testVec, testLabel = readData(args.testData);
testVec = np.array(testVec)
testLabel = np.array(testLabel).reshape(-1,1)

vec, label = readData(args.trainData)
vec = np.array(vec)
label = np.array(label)

# Create and fit an AdaBoosted decision tree
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=args.trails)
bdt.fit(vec, label)
for t in range(0,len(testVec)):
	if(bdt.predict(testVec[t].reshape(1,-1)) != testLabel[t]):
		boostErr += 1

if(args.verbose):
	print(bdt.N, len(testVec), args.trails, boostErr/len(testVec) )
else:
	print(args.trails, boostErr/len(testVec)) 	

