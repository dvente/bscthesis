#!/usr/bin/env python3.4
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import argparse
from sklearn import tree
import matplotlib.pyplot as plt
import subprocess


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
		self.sDiv = np.zeros((self.T,self.N))
		self.p = np.array((1,self.N))
		self.gamma = np.zeros(self.T)
		self.weakLearn = np.array([DecisionTreeClassifier(max_depth = 1) for i in range(0,self.T)])

	def fit(self):
		for t in range(1,self.T):
			# self.negMin = vNeg(self.s[t-1]-1)
			# self.negMax = vNeg(self.s[t-1]+1)
			# self.negMaxSq = np.power(self.negMax,2)
			# self.negMinSq = np.power(self.negMin,2)
			# self.negMaxSqDiv = self.negMax/(3*t)
			# self.negMinSqDiv = self.negMin/(3*t)
			# #print(self.negMinSqDiv, self.negMaxSqDiv)
			# self.negMaxSqDivExp = np.exp(self.negMaxSqDiv)
			# self.negMinSqDivExp = np.exp(self.negMinSqDiv)
			
			#print(self.negMax)
			#	raise ValueError("vNeg not working")
			self.p = np.exp(np.square(vNeg(self.s[t-1]-1))/(3*t))-np.exp(np.square(vNeg(self.s[t-1]+1))/(3*t))
			#self.p = self.negMinSqDivExp - self.negMaxSqDivExp
			self.p = self.p/sum(self.p)
			if(not isDist(self.p)):
				print(sum(list(map(lambda x: x<0,self.p))))
				print(np.isclose(sum(self.p),1.0))
				raise ValueError("Non distribution found")
				
			self.weakLearn[t].fit(self.vec, self.label.reshape(-1,1), sample_weight=self.p)#fit weakLearn with dist

			self.gamma[t] = 0.5*sum(self.p*self.label*self.weakLearn[t].predict(self.vec))#calc edge of hypothesis
			if(self.gamma[t] < 0 ):
				raise ValueError("Invalid edge")
			self.s[t] = self.s[t-1]+((0.5*self.label*self.weakLearn[t].predict(self.vec))-self.gamma[t])


	def predict(self,vec):
		predict = [self.weakLearn[t].predict(vec) for t in range(1,self.T)]
		ans = sum(predict)
		return np.sign(ans)

parser = argparse.ArgumentParser(description='NHBoostDT trainer')
parser.add_argument('trails', metavar = 'T', type = int, help='Number of trails to train NHBoostDT with')
parser.add_argument('--trainData', default = "../generated/DTtrain.dat",help="location of the training data")
parser.add_argument("--testData", default = "../generated/DTtest.dat", help = "location of the test data" )
parser.add_argument("-c", "--clean", action = "store_true")
parser.add_argument("-v", "--verbose", help="display number of traing and test cases aswell", action="store_true")

args = parser.parse_args()

if(args.clean):
	with open(args.trainData, 'r') as f:
		first_line = f.readline()
	
	N = int(first_line.rstrip())
	
	with open(args.testData, 'r') as f:
		first_line = f.readline()

	M = int(first_line.rstrip())
	subprocess.call(["python3.4 generate.py " + str(N) + " " + args.trainData + " -l 1" ], shell=True)
	subprocess.call(["python3.4 generate.py " + str(M) + " " + args.testData+ " -l 1" ], shell=True)

boostErr = 0
incl = 0
a = NHBoostDT(args.trails, args.trainData)
a.fit()

#print(np.sign(sum([a.weakLearn[t].predict(a.vec[1].reshape(1,-1)) for t in range(1,a.T)])))

#output format of a.predict is incorrect i.e. [0] 

testVec, testLabel = readData(args.testData);
testVec = np.array(testVec)
testLabel = np.array(testLabel)
for t in range(0,len(testVec)):
	ans = a.predict(testVec[t].reshape(1,-1))
	if(ans == [0]):
		incl += 1
	#if(a.predict(testVec[t].reshape(1,-1)) == [0]):
		# print(testVec[t].reshape(1,-1))
		# print(a.predict(testVec[t].reshape(1,-1)))
		# print(sum(map(lambda x: x**2, testVec[t])) > 9.34)
	if(ans != testLabel[t]):
		#print(a.predict(testVec[t].reshape(1,-1)))
		#print(testLabel[t])
		boostErr += 1

if(args.verbose):
	print(a.N, len(testVec), args.trails, boostErr/len(testVec) )
else:
	print( args.trails, boostErr/len(testVec), incl/len(testVec),(len(a.p)-np.count_nonzero(a.p))/len(a.p) )	
# col = a.sDiv[:,5]
# print(col)
# plt.plot(col)
# plt.show()
# slice = 24
# print(a.s[:,slice])
# print(a.sDiv[:,slice])