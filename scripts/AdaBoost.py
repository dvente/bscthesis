#!/usr/bin/python3
import numpy as np
import sys
from sklearn.tree import DecisionTreeClassifier

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
		vec.append(list(map(float, array[:-1])))
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
			self.p = self.w/sum(self.w)#normalize to dist
			if(not isDist(self.p)):
				print(sum(list(map(lambda x: x<0,self.p))))
				print(np.isclose(sum(self.p),1.0))
				raise ValueError("Non distribution found")
				
			self.weakLearn[t].fit(self.vec, self.label.reshape(-1,1), sample_weight=self.p)#fit weakLearn with dist
		#print(self.weakLearn[t].predict(self.vec))
			self.err = sum(np.multiply(self.p,abs(self.weakLearn[t].predict(self.vec)-self.label)))#calc err on training set
		#	print(self.err)
			self.beta[t] = self.err/(1-self.err)#set beta
			
			self.w = np.multiply(self.w,pow(self.beta[t], 1-abs(self.weakLearn[t].predict(self.vec)-self.label)))#update weights

		self.thresh = 0.5*sum(np.log(1/self.beta))

	def predict(self,vec):
		height = 0
		for t in range(0,self.T):
			height += np.log(1/self.beta[t])*self.weakLearn[t].predict(vec.reshape(1,-1))
		if height >= self.thresh:
			return 1
		else:
			return 0

def conv(inp):
	if inp:
		return 1
	else:
		return 0

# #stumpErr = 0
test = 10000
boostErr = 0
a = AdaBoost(int(sys.argv[1]), "train.dat", uniformDist)
a.fit()

s = DecisionTreeClassifier(max_depth = 1)
s.fit(a.vec, a.label, sample_weight = a.p)


for t in range(0,test):
	vec = np.random.normal(0,1,10)
	true = conv((sum(map(lambda x: x**2, vec)) > 9.34))
	if a.predict(vec) != true:
		boostErr += 1

print( sys.argv[1], boostErr/test) 	

