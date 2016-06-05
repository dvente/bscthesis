import numpy as np
from sklearn.tree import DecisionTreeClassifier


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
		label.append(float(array[-1]))

	return (vec,label)

class DecisionStump():
	def __init__(self):
		self.dimen = -1
		self.threshVal = float('inf')
		self.threshIneq = 'lt'
		self.minLabel = 0
		self.maxLabel = 1

	def predict(self,dataMatrix):
		retArray = np.ones(dataMatrix.shape[0])
		retArray.fill(self.maxLabel)
		if self.threshIneq == 'lt':
			retArray[dataMatrix[:,self.dimen] <= self.threshVal] = self.minLabel
		else:
			retArray[dataMatrix[:,self.dimen] > self.threshVal] = self.minLabel
		return retArray


	def stumpClassify(self,dataMatrix,dimen,threshVal,threshIneq):
		retArray = np.ones((dataMatrix.shape[0],1))
		retArray.fill(self.maxLabel)
		if threshIneq == 'lt':
			retArray[dataMatrix[:,dimen] <= threshVal] = self.minLabel
		else:
			retArray[dataMatrix[:,dimen] > threshVal] = self.minLabel
		return retArray
	

	def fit(self,dataArr,classLabels,sample_weight):
		dataMatrix = np.array(dataArr); labelMat = np.array(classLabels).reshape(-1,1)
		m,n = dataMatrix.shape
		self.minLabel = int(min(labelMat))
		self.maxLabel = int(max(labelMat))
		numSteps = 100 #bestStump = {}; bestClasEst = np.zeros((m,1))
		minError = float('inf')     
		for i in range(n):
			rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max();
			stepSize = (rangeMax-rangeMin)/numSteps
			for j in range(-1,int(numSteps)+1):
				for inequal in ['lt', 'gt']: 
					threshVal = (rangeMin + float(j) * stepSize)
					predictedVals = self.stumpClassify(dataMatrix,i,threshVal,inequal)
					errArr = np.ones((m,1))
					errArr[predictedVals == labelMat] = 0
					weightedError = np.dot(sample_weight.T,errArr)  
					#print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError))
					if weightedError < minError:
						minError = weightedError
						#bestClasEst = predictedVals.copy()
						self.dimen = i
						self.threshVal = threshVal
						self.threshIneq = inequal
					if weightedError == 0: #perfect fit so stop
						return


# testVec, testLabel = readData("stumpTest.dat");
# testVec = np.array(testVec)
# testLabel = np.array(testLabel).reshape(-1,1)
# D = np.array([1,1,1,1,1,1,1,1,1,1])
# D = D / sum(D)
# s = DecisionStump()
# s.fit(testVec,testLabel,D)
# print(testLabel.reshape(1,-1))
# print(s.predict(testVec))

# w = DecisionTreeClassifier(max_depth=1)
# w.fit(testVec, testLabel, sample_weight = D)
# print(w.predict(testVec))