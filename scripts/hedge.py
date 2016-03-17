import numpy as np
import random as rand
import string

def uniformDist(N):
	w = [1/float(N) for i in range(0,N)]
	return w

def isDist(w):
		if(not np.isclose(sum(w),1.0)):
			raise ValueError('Distribution does not sum to 1')
		elif(sum(map(lambda x : x > 0.0 and x < 1.0, w)) != len(w)):
			raise ValueError('Distribution out of range')
		else:
			return True	

class hedgeL:

	def __init__(self, T, w, loss, beta = None):
		self.T = T
		self.N = len(w) 
		self.loss = loss
		self.cumLoss = 0
		self.p = [0.0 for j in range(0,self.N)]
		self.lossVec = [0.0 for j in range(0,self.N)]#dist and loss vector
		self.copy = [] #list to store w[t-1] while calcing w[t]

		if isDist(w):#errors are handeld by isDist
			self.w = w #set initial dist

		if beta == None:# from paper, is this correct?
			self.beta = 1/float(1+np.sqrt(2/float(self.T/float(np.log(self.N)))))
		else:
			self.beta = beta

	def train(self):
		for t in range(0,self.T):
			for i in range(0, self.N):#choose weights
				self.p[i] = float(self.w[i])/float(sum(self.w))
			
			self.lossVec = self.loss(self.N) #recieve loss vector
			self.cumLoss += sum([self.p[i]*self.w[i] for i in range(0,self.N)]) #log the losses
			self.copy = self.w
			
			for i in range(0,self.N): #reassign wieghts
				self.w[i] = self.copy[i]*np.power(self.beta,self.lossVec[i])

#Delta = Lambda = [list(string.ascii_uppercase)[0:15]]
def kPredict(N):
	loss = [1 for i in range(0,N)]
	loss[list(string.ascii_uppercase).index(rand.choice(population))] = 0
	return loss

choices =[(letter,rand.randint(1,5)) for letter in list(string.ascii_uppercase)[0:10]]
choices[0] = ('A', 20)
population = [val for val, cnt in choices for i in range(cnt)]
