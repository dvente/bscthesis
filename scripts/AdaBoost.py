#!/usr/bin/python3
import numpy as np
import sys
from sklearn.tree import DecisionTreeClassifier
import pandas

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


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
		vec.append(list(map(float, array[:-2])))
		label.append(int(array[-1]))
	
	return (vec,label)

class AdaBoost:
	def __init__(self, T, data, dataDist):
		self.T = T 
		self.vec, self.label = readData(data)
		self.vec = np.array(self.vec)
		self.label = np.array(self.label)
		self.N = len(self.vec)
		self.w = np.array(dataDist(self.N))
		self.p = np.array(self.w/sum(self.w))
		self.err = 0
		self.beta = np.zeros(self.T)
		self.weakLearn = np.array([DecisionTreeClassifier(max_depth = 1) for i in range(0,self.T)])
		self.thresh = 0

	def fit(self):
		for t in range(0,self.T):
			self.p = self.w/sum(self.w)#normalize
			self.weakLearn[t].fit(self.vec, self.label, sample_weight=self.p)
			self.err = sum(self.p*abs(self.weakLearn[t].predict(self.vec)-self.label))
			self.beta[t] = self.err/(1-self.err)
			self.w = self.w*pow(self.beta[t], 1-abs(self.weakLearn[t].predict(self.vec)-self.label))

		self.thresh = 0.5*sum(np.log(1/self.beta))

	def predict(self,vec):
		height = 0
		for t in range(0,self.T):
			height += np.log(1/self.beta[t])*self.weakLearn[t].predict(vec)
		if height >= self.thresh:
			return 1
		else:
			return -1

def conv(inp):
	if inp:
		return 1
	else:
		return -1

# #stumpErr = 0
boostErr = 0
test = 5000
a = AdaBoost(int(sys.argv[1]), "../generated/train.dat", uniformDist)
a.fit()


	
for t in range(0,test):
	vec = np.random.normal(0,1,9).tolist()
	print(vec)
	print(list(map(lambda x: x**2, vec)))
	print(sum(list(map(lambda x: x**2, vec))))
	true = conv((sum(map(lambda x: x**2, vec)) > 9.34))
	if a.predict(vec) != true:
		boostErr += 1
	print(a.predict(vec), conv(true))



print( sys.argv[1], boostErr/test) 	

