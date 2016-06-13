import numpy as np
import sys
from sklearn.tree import DecisionTreeClassifier
import argparse
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def isDist(array):
	return (sum(list(map(lambda x: x<0,array)))==0 and np.isclose(sum(array),1.0))

def uniformDist(N):
	return  [float(1/N) for i in range(0,N)]

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

class AdaBoost:
	def __init__(self, T, data, dataDist):
		self.T = T 
		self.vec, self.label = readData(data)
		#self.vec, self.label = load_svmlight_file("a9a0")
		self.vec = np.array(self.vec)
		self.label = np.array(self.label)
		self.N = self.vec.shape[0]
		self.w = np.array(dataDist(self.N))
		self.p = np.array(self.w/sum(self.w))
		self.err = 0
		self.beta = np.zeros((self.T,1))
		self.weakLearn = np.array([DecisionTreeClassifier(max_depth = 1) for i in range(0,self.T)])
		#self.weakLearn = np.array([DecisionStump() for i in range(0,self.T)])
		self.thresh = 0
		self.trainError = np.zeros((self.T,1))

	def fit(self):
		for t in range(0,self.T):
			self.p = self.w/sum(self.w)#normalize to dist
			if(not isDist(self.p)):
				print(sum(list(map(lambda x: x<0,self.p))))
				print(np.isclose(sum(self.p),1.0))
				print(sum(self.p))
				raise ValueError("Non distribution found")
				
			self.weakLearn[t].fit(self.vec, self.label, sample_weight=self.p)#fit weakLearn with dist

			self.err = sum(np.multiply(self.p,abs(self.weakLearn[t].predict(self.vec)-self.label)))#calc err on training set

			self.beta[t] = self.err/(1-self.err)
			
			self.w = np.multiply(self.w,pow(self.beta[t], 1-abs(self.weakLearn[t].predict(self.vec)-self.label)))#update weights
			self.trainError[t] = np.sum(self.p*np.abs(self.weakLearn[t].predict(self.vec)-self.label)*0.5)

		self.thresh = 0.5*sum(np.log(1/self.beta))

	def predict(self,vec):
		height = 0
		for t in range(0,self.T):
			height += np.log(1/self.beta[t])*self.weakLearn[t].predict(vec)
		if height >= self.thresh:
			return 1
		else:
			return 0

def conv(inp):
	if inp:
		return 1
	else:
		return 0

parser = argparse.ArgumentParser(description='AdaBoost trainer')
parser.add_argument('trails', metavar = 'T', type = int, help='Number of trails to train AdaBoost with')
parser.add_argument('--trainData', default = "../data/AdaTrain.dat",help="location of the training data")
parser.add_argument("--testData", default = "../data/AdaTest.dat", help = "location of the test data" )
parser.add_argument("--log", help="log answer to file instead of returning" )
parser.add_argument("-v", "--verbose", help="display number of traing and test cases aswell", action="store_true")

args = parser.parse_args()

boostErr = 0
a = AdaBoost(args.trails, args.trainData, uniformDist)
a.fit()
# print("done fitting")
testVec, testLabel = readData(args.testData)  #load_svmlight_file("a9a0.t")
for t in range(0,len(testLabel)):
	ans = a.predict(testVec[t])
	if(ans != testLabel[t]):
		boostErr += 1

if(args.log):
	with open(args.log,'a') as file:
		file.write(str(a.T) + " " + str(boostErr/len(testLabel)) + " 0.0 0.0 \n")
else:
	print( a.T, boostErr/len(testLabel), 0.0, 0.0) 	

