import numpy as np
import sys
from sklearn.tree import DecisionTreeClassifier
import argparse
from sklearn.datasets import load_svmlight_file
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def isDist(array):
	return (np.all(array>0) and np.isclose(sum(array),1.0))

def uniformDist(N):
	return np.array([float(1/N) for i in range(0,N)])

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


parser = argparse.ArgumentParser(description='AdaBoost trainer')
parser.add_argument('-n', '--nodes', type = int, default = 0, help='Maximum number of nodes of the tree, use 0 for unlimited nodes.')
parser.add_argument('--svm', action="store_true")
parser.add_argument('--trainData', default = "../generated/train.dat",help="location of the training data")
parser.add_argument("--testData", default = "../generated/test.dat", help = "location of the test data" )
parser.add_argument("-v", "--verbose", help="display number of traing and test cases aswell", action="store_true")
args = parser.parse_args()

boostErr = 0
if args.svm:
	vec, label = load_svmlight_file("a9a")
else:
	vec, label = readData(args.trainData)
	vec = np.array(vec)
	label = np.array(label)


if(args.nodes == 0):
	a = DecisionTreeClassifier()
else:
	a = DecisionTreeClassifier(max_depth = max(np.log2(args.nodes),1))

a.fit(vec,label)

if args.svm:
	testVec, testLabel = load_svmlight_file("a9a.t")
else:
	testVec, testLabel = readData(args.testData);
	testVec = np.array(testVec)
	testLabel = np.array(testLabel)

for t in range(0,len(testLabel)):
	if(a.predict(testVec[t]) != testLabel[t]):
		boostErr += 1

if(args.verbose):
	print(a.N, len(testLabel), a.tree_.node_count, boostErr/len(testLabel) )
else:
	print(a.tree_.node_count, boostErr/len(testLabel)) 	


