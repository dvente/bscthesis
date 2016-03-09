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
		self.vec = np.asarray(self.vec)
		self.label = np.asarray(self.label)
		self.N = len(self.vec)
		self.err = np.zeros(self.T)#[0 for i in range(0,self.T)]
		self.w = [dataDist(len(self.vec)) for t in range(0,self.T+1)]
		self.p = np.zeros((self.T,self.N))
		self.beta = np.zeros(self.T)
		self.weakLearn = [DecisionTreeClassifier(max_depth = 1) for i in range(0,self.T)]
		self.train()
		self.thresh = 0
		for t in range(0,self.T):
			self.thresh += (np.log(1/self.beta[t]))
		self.thresh *= 0.5


	def train(self):
		for t in range(0,self.T):
			for i in range(0, self.N):

				self.p[t][i] = float(self.w[t][i])/max(float(sum(self.w[t])),0.00000000001)
			#print(self.p[t].shape)	
			
			#print(self.p[t])
			self.weakLearn[t].fit(self.vec, self.label.reshape(-1,1), sample_weight=self.p[t])

			for i in range(0, self.N):
				self.err[t] += self.p[t][i]*abs(self.weakLearn[t].predict(self.vec[i])-self.label[i])

			self.beta[t] = self.err[t]/float(1-self.err[t])

			for i in range(0, self.N):#choose weights
				self.w[t+1][i] = self.w[t][i]*pow(self.beta[t],1-abs(self.weakLearn[t].predict(self.vec[i])-self.label[i]))

	def finalHyp(self, vec):
		height = 0
		for t in range(0,self.T):
			height += (np.log(1/self.beta[t]))*self.weakLearn[t].predict(vec)
			
		if height > self.thresh:
			return 1
		else:
			return -1


sumpErr = 0
boostErr = 0
test = 1000
a = AdaBoost(int(sys.argv[1]), "../generated/train.dat", uniformDist)
s = DecisionTreeClassifier(max_depth = 1)
s.fit(a.vec, a.label.reshape(-1,1))
	
for t in range(0,test):
 	vec = np.random.normal(0,1,9).tolist()
 	true = (sum(list(map(lambda x: x**2, vec))) > 9.34)
 	boostErr += abs(a.finalHyp(vec)-true)
 	sumpErr += abs(s.predict(vec)-true)
print()
print("trails: " + sys.argv[1])
print("test cases: " +str(test))
print("train cases: " + str(a.N))
print("boostErr:" + str(boostErr))
print("sumpErr:" + str(sumpErr))
# print float(guess)/float(1000)
