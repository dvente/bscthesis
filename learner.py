import numpy as np

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
		
		#self.loss = loss
		self.T = T
		self.N = len(w) 
		self.loss = loss
		self.cumLoss = 0
		self.p = [0.0 for j in range(0,self.N)]#dist vector
		self.copy = [] #list to store w[t-1] while calcing w[t]

		if isDist(w):#errors are handeld by isDist
			self.w = w #set initial dist

		if beta == None:# from paper, is this correct?
			self.beta = 1/float(1+np.sqrt(2/float(self.T/float(np.log(self.N)))))
		else:
			self.beta = beta


h = hedgeL(10, uniformDist(10))
print h.w
