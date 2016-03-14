#!/usr/bin/python3.4
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import sys
import pandas

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def readData(file):
	vec = []
	label = []
	f = open(file,'r')
	for line in f:
		line = line.rstrip()
		array = line.split(" ")
		vec.append(list(map(float, array[:-2])))
		label.append(int(array[-1]))
	
	return (vec,label)

vec, label = readData("../generated/train.dat")

# Create and fit an AdaBoosted decision tree
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=int(sys.argv[1]))
bdt.fit(vec, label)


sumpErr = 0
boostErr = 0
test = 1000
#a = AdaBoost(int(sys.argv[1]), "../generated/train.dat", uniformDist)
# s = DecisionTreeClassifier(max_depth = 1)
# s.fit(vec, label)
	
# for t in range(0,test):
#  	inp = np.random.normal(0,1,9).tolist()
#  	true = (sum(list(map(lambda x: x**2, inp))) > 9.34)
#  	boostErr += abs(bdt.predict(inp)-true)
#  	sumpErr += abs(s.predict(inp)-true)
# print()
# print("trails: " + sys.argv[1])
# print("test cases: " +str(test))
# print("train cases: " + str(len(vec)))
# print("boostErr:" + str(boostErr))
# print("sumpErr:" + str(sumpErr))
# print float(guess)/float(1000)

def conv(inp):
	if inp:
		return [1]
	else:
		return [-1]

for t in range(0,test):
	vec = np.random.normal(0,1,9).tolist()
	true = conv((sum(map(lambda x: x**2, vec)) > 9.34))
#	print(true, bdt.predict(vec))
	if bdt.predict(vec) != true:
# 		print(a.predict(vec))
		boostErr += 1


print( sys.argv[1], boostErr/test ) 
