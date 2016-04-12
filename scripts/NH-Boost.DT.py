#!/usr/bin/python3.4
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import argparse
from sklearn.externals.six import StringIO
from sklearn import tree


def isDist(array):
	return (sum(list(map(lambda x: x<0,array)))==0 and np.isclose(sum(array),1.0))

def uniformDist(N):
	w = [float(1/N) for i in range(0,N)]
	return w

def neg(s):
	return min(0,s)

vNeg = np.vectorize(neg)
	
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

class NHBoostDT:
	def __init__(self, T, data):
		self.T = T + 1
		self.vec, self.label = readData(data)
		self.vec = np.array(self.vec)
		self.label = np.array(self.label)
		self.N = len(self.vec)
		self.s = np.zeros((self.T,self.N))
		self.p = np.array((1,self.N))
		self.gamma = 0
		self.weakLearn = np.array([DecisionTreeClassifier(max_depth = 1) for i in range(0,self.T)])

	def fit(self):
		for t in range(1,self.T):
			self.p = np.exp(pow(vNeg(self.s[t-1]-1),2)/(3*t))-np.exp(pow(vNeg(self.s[t-1]+1),2)/(3*t))
			self.p = self.p/sum(self.p)
			if(not isDist(self.p)):
				print(sum(list(map(lambda x: x<0,self.p))))
				print(np.isclose(sum(self.p),1.0))
				raise ValueError("Non distribution found")
				
			self.weakLearn[t].fit(self.vec, self.label.reshape(-1,1), sample_weight=self.p)#fit weakLearn with dist

			self.gamma = 0.5*sum(self.p*self.label*self.weakLearn[t].predict(self.vec))#calc err on training set
			
			self.s[t] = self.s[t-1]+0.5*self.label*self.weakLearn[t].predict(self.vec)-self.gamma

	def predict(self,vec):
		predict = [self.weakLearn[t].predict(vec) for t in range(1,self.T)]
		ans = sum(predict)
		if (ans == [0]):
			return np.sign(ans + predict[1])#remove first predictor to break the tie
		else:
			return np.sign(ans)

parser = argparse.ArgumentParser(description='NHBoostDT trainer')
parser.add_argument('trails', metavar = 'T', type = int, help='Number of trails to train NHBoostDT with')
parser.add_argument('--trainData', default = "../generated/DTtrain.dat",help="location of the training data")
parser.add_argument("--testData", default = "../generated/DTtest.dat", help = "location of the test data" )
parser.add_argument("-v", "--verbose", help="display number of traing and test cases aswell", action="store_true")

args = parser.parse_args()

boostErr = 0
a = NHBoostDT(args.trails, args.trainData)
a.fit()

#print(np.sign(sum([a.weakLearn[t].predict(a.vec[1].reshape(1,-1)) for t in range(1,a.T)])))

#output format of a.predict is incorrect i.e. [0] 

testVec, testLabel = readData(args.testData);
testVec = np.array(testVec)
testLabel = np.array(testLabel)
for t in range(0,len(testVec)):
	#if(a.predict(testVec[t].reshape(1,-1)) == [0]):
		# print(testVec[t].reshape(1,-1))
		# print(a.predict(testVec[t].reshape(1,-1)))
		# print(sum(map(lambda x: x**2, testVec[t])) > 9.34)
	if(a.predict(testVec[t].reshape(1,-1)) != testLabel[t]):
		#print(a.predict(testVec[t].reshape(1,-1)))
		#print(testLabel[t])
		boostErr += 1

if(args.verbose):
	print(a.N, len(testVec), args.trails, boostErr/len(testVec) )
else:
	print( args.trails, boostErr/len(testVec)) 	


