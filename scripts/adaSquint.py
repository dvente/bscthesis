#!/usr/bin/python3.5
import numpy as np
import sys
from sklearn.tree import DecisionTreeClassifier
import argparse
import matplotlib.pyplot as plt
import squint


def pos(s):
	return max(0,s)

vPos = np.vectorize(pos)

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
		self.vec = np.array(self.vec)
		self.label = np.array(self.label)
		self.N = len(self.vec)
		self.w = np.array(dataDist(self.N))
		self.p = np.array(self.w/sum(self.w))
		self.err = 0
		self.beta = np.zeros(self.T)
		self.weakLearn = np.array([DecisionTreeClassifier(max_depth = 1) for i in range(0,self.T)])
		self.thresh = 0
		self.trainError = np.zeros((self.T,1))
		self.R = np.zeros((self.T,self.N))
		self.R_normal = np.zeros((self.T,self.N))

	def fit(self):
		for t in range(0,self.T):
			self.p = self.w/sum(self.w)#normalize to dist
			if(not isDist(self.p)):
				print(sum(list(map(lambda x: x<0,self.p))))
				print(np.isclose(sum(self.p),1.0))
				print(sum(self.p))
				raise ValueError("Non distribution found")
			
			if t > 0 :
				p_normal = np.exp(np.square(vPos(self.R_normal[t-1]-1))/(3*t))-np.exp(np.square(vPos(self.R_normal[t-1]+1))/(3*t))	
				p_normal = p_normal / sum(p_normal)
			else:			
				p_normal = np.array([1/self.N for i in range(self.N)])
				
			self.weakLearn[t].fit(self.vec, self.label.reshape(-1,1), sample_weight=p_normal)#fit weakLearn with dist

			self.err = sum(np.multiply(self.p,abs(self.weakLearn[t].predict(self.vec)-self.label)))#calc err on training set

			self.beta[t] = self.err/(1-self.err)#set beta
			
			self.w = np.multiply(self.w,pow(self.beta[t], 1-abs(self.weakLearn[t].predict(self.vec)-self.label)))#update weights

			r = 0.5*((np.abs(self.weakLearn[t].predict(self.vec)-self.label)) - sum(np.abs(self.weakLearn[t].predict(self.vec)-self.label)*self.p))
			r_normal =  0.5*((np.abs(self.weakLearn[t].predict(self.vec)-self.label)) - sum(np.abs(self.weakLearn[t].predict(self.vec)-self.label)*p_normal))
			
			if t > 0:
				self.R[t] = self.R[t-1] + r
				self.R_normal[t] = self.R_normal[t-1] + r_normal
			else:
				self.R[t] = r
				self.R_normal[t] = r_normal

			# self.trainError[t] = np.sum(self.p*np.abs(self.weakLearn[t].predict(self.vec)-self.label)*0.5)

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

parser = argparse.ArgumentParser(description='AdaBoost trainer')
parser.add_argument('trails', metavar = 'T', type = int, help='Number of trails to train AdaBoost with')
parser.add_argument('--trainData', default = "../generated/train.dat",help="location of the training data")
parser.add_argument("--testData", default = "../generated/test.dat", help = "location of the test data" )
parser.add_argument("-v", "--verbose", help="display number of traing and test cases aswell", action="store_true")

args = parser.parse_args()

boostErr = 0
a = AdaBoost(args.trails, args.trainData, uniformDist)
a.fit()

testVec, testLabel = readData(args.testData);
testVec = np.array(testVec)
testLabel = np.array(testLabel)
for t in range(0,len(testVec)):
	if(a.predict(testVec[t].reshape(1,-1)) != testLabel[t]):
		boostErr += 1

plt.plot(range(a.T), [max(a.R[t]) for t in range(a.T)], color='r')
plt.grid()
plt.show()

plt.plot(range(a.T), [max(a.R_normal[t]) for t in range(a.T)], color='r')
plt.grid()
plt.show()

if(args.verbose):
	print(a.N, len(testVec), args.trails, boostErr/len(testVec) )
else:
	print( args.trails, boostErr/len(testVec)) 	

