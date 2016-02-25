import numpy as np
import sys
import pandas
from sklearn.neighbors import KNeighborsClassifier

def uniformDist(N):
	w = [1/float(N) for i in range(0,N)]
	return w

def readData(file):
	vec = []
	label = []
	f = open(file,'r')
	for line in f:
		line = line.rstrip()
		array = line.split(" ")
		vec.append(map(float, array[:-2]))
		label.append(int(array[-1]))
	
	return (vec,label)

class AdaBoost:
	def __init__(self, T, data, dataDist, weakLearn):
		self.T = T
		#self.data = 
		self.vec, self.label = readData(data)#dataDist(len(data))
		self.hyp = weakLearn
		self.N = len(self.vec)
		self.err = [0 for i in range(0,self.T)]
		self.w = [dataDist(len(self.vec)) for t in range(0,self.T+1)]
		self.p = [[0 for i in range(0,self.N)] for j in range(0,self.T+1)]
		self.beta = [0 for i in range(0,self.T)]
		self.train()


	def train(self):
		for t in range(0,self.T):
			for i in range(0, self.N):
				self.p[t][i] = float(self.w[t][i])/float(sum(self.w[t]))
				self.err[t] += self.p[t][i]*abs(self.hyp(self.vec[i])-self.label[i])

			self.beta[t] = self.err[t]/float(1-self.err[t])

			for i in range(0, self.N):#choose weights
				self.w[t+1][i] = self.w[t][i]*pow(self.beta[t],1-abs(self.hyp(self.vec[i])-self.label[i]))

	def finalHyp(self, vec):
		height = 0
		thresh = 0
		for t in range(0,self.T):
			height += (log(1/float(beta[t])))*self.hyp(vec)
			thresh += (log(1/float(beta[t])))
		if height > 0.5*thresh:
			return 1
		else:
			return 0


# class weakLearn:
# 	def __init__(self, data):
# 		self.vec, self.label = readData(data)
# 		self.kNN = KNeighborsClassifier(n_neighbors = 15, n_jobs = -1)
# 		#print self.vec
# 		#print self.label
# 		self.kNN.fit(self.vec, self.label)

# 	def algo(self, vec):
# 		print np.random.normal(0,1,1)
# 		return self.kNN.predict(vec)

def stump(vec):
	if sum(vec) > 9.34:
		return 1
	else:
		return 0

# true = (sum(map(lambda x: x**2, vec)) > 9.34)
# guess = a.finalHyp(vec)
# print guess
# print true
guess = 0
a = AdaBoost(100, "train.dat", uniformDist, stump)
for t in range(0,100):
 	vec = np.random.normal(0,1,10).tolist()
  	true = (sum(map(lambda x: x**2, vec)) > 9.34)
  	guess += abs(a.finalHyp(vec)-true)
print guess
# print float(guess)/float(1000)
