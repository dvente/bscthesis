import numpy as np
from sklearn.tree import DecisionTreeClassifier
from stump import DecisionStump
import argparse
from sklearn import tree
import subprocess
import warnings
from sklearn.datasets import load_svmlight_file
warnings.filterwarnings("ignore", category=DeprecationWarning)



parser = argparse.ArgumentParser(description='NHBoostDT trainer')
parser.add_argument('trails', metavar = 'T', type = int, help='Number of trails to train NHBoostDT with')
parser.add_argument('--trainData', default = "../generated/DTtrain.dat",help="location of the training data")
parser.add_argument("--testData", default = "../generated/DTtest.dat", help = "location of the test data" )
parser.add_argument("-c", "--clean", action = "store_true", help="generate new data files before executing")

args = parser.parse_args()

def isDist(array):
	return (sum(list(map(lambda x: x<0,array)))==0 and np.isclose(sum(array),1.0))

def pos(s):
	return max(0,s)

def neg(s):
  return min(0,s)   

vPos = np.vectorize(pos)
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

testVec, testLabel = readData(args.testData);
testVec = np.array(testVec)
testLabel = np.array(testLabel)

class NHBoostDT:
	def __init__(self, T, data):
		self.T = T + 1
		self.vec, self.label = load_svmlight_file("a9a")#readData(data)
		# self.vec = np.array(self.vec)
		# self.label = np.array(self.label)
		self.N = self.vec.shape[0]
		self.s = np.zeros(self.N)
		self.p = np.array((1,self.N))
		#self.gamma = np.zeros(self.T)
		self.weakLearn = np.array([DecisionTreeClassifier(max_depth = 1) for i in range(0,self.T)])
		#self.weakLearn = np.array([DecisionStump() for i in range(0,self.T)])

	def fit(self):
		for t in range(1,self.T):
			self.p = np.exp(np.square(vNeg(self.s-1))/(3*t))-np.exp(np.square(vNeg(self.s+1))/(3*t))
			self.p = self.p / sum(self.p)
			if(not isDist(self.p)):
				raise ValueError("non dist")
			self.weakLearn[t].fit(self.vec, self.label, sample_weight = self.p)
			self.gamma = np.sum(self.p*self.label*self.weakLearn[t].predict(self.vec))/2
			#print((self.label*self.weakLearn[t].predict(self.vec)).shape)
      # negated regret:
			self.s = self.s + self.label*self.weakLearn[t].predict(self.vec) - self.gamma



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

boostErr = 0
incl = 0
a = NHBoostDT(args.trails, args.trainData)
a.fit()

 
testVec, testLabel = load_svmlight_file("a9a.t")
# testVec = np.array(testVec)
# testLabel = np.array(testLabel)
for t in range(0,len(testLabel)):
	ans = a.predict(testVec[t])
	if(ans == [0]):
		incl += 1
		boostErr += 0.5
	elif(ans != testLabel[t]):
		boostErr += 1

print( args.trails, boostErr/len(testLabel), incl/len(testLabel),(len(a.p)-np.count_nonzero(a.p))/len(a.p) )	
