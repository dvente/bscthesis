import numpy as np
from sklearn.tree import DecisionTreeClassifier
from stump import DecisionStump
import argparse
import squint
import subprocess
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)



parser = argparse.ArgumentParser(description='SquintBoost trainer')
#parser.add_argument('trails', metavar = 'T', type = int, help='Number of trails to train NHBoostDT with')
parser.add_argument('--trainData', default = "../generated/SQtrain.dat",help="location of the training data")
parser.add_argument("--testData", default = "../generated/SQtest.dat", help = "location of the test data" )
parser.add_argument("-c", "--clean", action = "store_true", help="generate new data files before executing")


args = parser.parse_args()

def isDist(array):
	return (sum(list(map(lambda x: x<0,array)))==0 and np.isclose(sum(array),1.0))

def uniformDist(N):
	return  [float(1/N) for i in range(0,N)]

def neg(s):
	return min(0,s)

vNeg = np.vectorize(neg)
vecEvidence = np.vectorize(squint.lnevidence)
	
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


class SquintBoost:
	def __init__(self, data):
		#self.T = T + 1
		self.T =  1000000
		self.vec, self.label = load_svmlight_file("a9a")
		# self.vec = np.array(self.vec)
		# self.label = np.array(self.label)
		self.N = len(self.label)
		self.p = np.array((1,self.N))
		self.weakLearn = np.array([DecisionTreeClassifier(max_depth = 1) for i in range(0,self.T)])
		self.lnpi = np.log(np.array(uniformDist(self.N)))
		#self.trainError = np.zeros((self.T,self.N))

	def fit(self):
		R = np.zeros(self.N)
		V = np.zeros(self.N)
		t = 0
		start = time.process_time()
		while time.process_time() - start < 60:
		#for t in range(1,self.T):
			lw = vecEvidence(R, V)
			w = np.exp(lw)
			self.p = w / sum(w) 

			self.weakLearn[t].fit(self.vec, self.label, sample_weight = self.p)
			margins = self.weakLearn[t].predict(self.vec)*self.label
			gamma = np.sum(self.p*self.weakLearn[t].predict(self.vec)*self.label)
			r = 0.5*(gamma-margins) 

			#vectorbased update
			R = R + r
			V = V + np.square(r)
			t += 1

		print("execution time: ", time.process_time() - start)
		print("iterations: " , t)
		self.T = t




	def predict(self,vec):
		predict = [self.weakLearn[t].predict(vec) for t in range(1,self.T)]
		ans = sum(predict)
		return np.sign(ans)

if(args.clean):
	with open(args.trainData, 'r') as f:
		first_line = f.readline()
	
	N = int(first_line.rstrip())
	
	with open(args.testData, 'r') as f:
		first_line = f.readline()

	M = int(first_line.rstrip())
	subprocess.call(["python generate.py " + str(N) + " " + args.trainData + " -l 1" ], shell=True)
	subprocess.call(["python generate.py " + str(M) + " " + args.testData+ " -l 1" ], shell=True)
print("Squint time")
boostErr = 0
incl = 0
a = SquintBoost(args.trainData)
a.fit()

testVec, testLabel = load_svmlight_file("a9a.t")
testStart = time.process_time()

for t in range(0,len(testLabel)):
	ans = a.predict(testVec[t])
	if(ans == [0]):
		incl += 1
		boostErr += 0.5
	elif(ans != testLabel[t]):
		boostErr += 1


print("testing time: ", time.process_time() - testStart)

print(a.T, boostErr/len(testLabel), incl/len(testLabel),(len(a.p)-np.count_nonzero(a.p))/len(a.p) )	
