import numpy as np
import numexpr as ne
import pandas

def uniformDist(N): 
	return np.array([1/float(N) for i in range(0,N)])



class AdaBoost:
	def __init__(self, data, trails, dist = uniformDist):
		df = pandas.read_csv(data)

		self.T = trails
		self.N = self.df.shape[0]
		self.p = np.array([0 for i in range(0,self.N)]) 
		self.w = [dist(self.N),dist(self.N)]
		self.b = np.array([0 for i in range(0,self.T)])
		self.train() 

	def train(self):
		for t in range(0,self.T):
			self.w[0] = self.w[1]
			sumW = np.sum(self.w[0])
			
			#vectorized verzion of p^t = w^t/(sum(w^t))
			self.p = ne.evaluate("w/sumW", local_dict = {'w': self.w[0], 'sumW': sumW })
			#TODO: split df into vecs and labels
			err = np.sum(ne.evaluate("p*abs(stump(h)-y)", local_dict = {'p':self.p, 'stump':stump, 'h':self.df.iloc, 'y':self.df.iloc[-1]})) 

			self.beta[t] = err/(1-err)

			self.w[1] = ne.evaluate('w*pow(beta,1-(abs(h-y)))', local_dict = {'w':self.w[0]})

def stump(vec,p):
	if sum(np.multiply(vec,p)) > 9.34:
		return 1
	else:
		return -1
# class AdaBoost:
# 	def __init__(self, T, data, dataDist, weakLearn):
# 		self.T = T
# 		self.thresh = 0
# 		self.vec, self.label = readData(data)#dataDist(len(data))
# 		self.hyp = weakLearn
# 		self.N = len(self.vec)
# 		self.err = [0 for i in range(0,self.T)]
# 		self.w = [dataDist(len(self.vec)) for t in range(0,self.T+1)]
# 		self.p = [[0 for i in range(0,self.N)] for j in range(0,self.T+1)]
# 		self.beta = [0 for i in range(0,self.T)]
# 		self.t = 0
# 		self.train()		

# 	def setDist(self, i):
# 		self.p[self.t][i] = float(self.w[self.t][i])/float(sum(self.w[self.t]))

# 	def setErr(self, i):
# 		self.err[self.t] += self.p[self.t][i]*abs(self.hyp(self.vec[i])-self.label[i])

# 	def train(self):
# 		pool = mult.Pool(processes = mult.cpu_count())
# 		for self.t in range(0,self.T):
# 			pool.map(self.setDist, range(0,self.N))
# 			pool.map(self.setErr, range(0,self.N))
# 			#for i in range(0,self.N):
# 				#self.p[t][i] = float(self.w[t][i])/float(sum(self.w[t]))
# 				#self.err[t] += self.p[t][i]*abs(self.hyp(self.vec[i])-self.label[i])


# 			self.beta[t] = self.err[t]/float(1-self.err[t])+1

# 			for i in range(0, self.N):
# 				self.w[t+1][i] = self.w[t][i]*pow(self.beta[t],1-abs(self.hyp(self.vec[i])-self.label[i]))
				
# 			self.thresh += (math.log(1/float(self.beta[t])))

# 		self.thresh *= 0.5

# 	def finalHyp(self, vec):
# 		height = 0
# 		for t in range(0,self.T):
# 			height += (math.log(1/float(self.beta[t])))*self.hyp(vec)
# 		if height > self.thresh:
# 			return 1
# 		else:
# 			return 0


# guess = 0
# a = AdaBoost(100, "train.dat", uniformDist, stump)
# for t in range(0,100):
#  	vec = np.random.normal(0,1,10).tolist()
#   	true = (sum(map(lambda x: x**2, vec)) > 9.34)
#   	guess += abs(a.finalHyp(vec)-true)
# print guess
# os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (0.1, 1000))
# # print float(guess)/float(1000)
#D = pandas.read_csv('train.dat')
#print D
a = AdaBoost('train.dat', 10)