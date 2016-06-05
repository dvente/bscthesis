import squint
import numpy as np
import matplotlib.pyplot as plt


class Squint:
	def __init__(self, pi, losses):
		self.T, self.N = losses.shape
		assert(np.array([(max(losses[:,i])-min(losses[:,i])) <= 1 for i in range(0,self.N)]).all())

		self.R = np.zeros((self.T,self.N))
		self.V = np.zeros((self.T,self.N))

		self.l_alg = np.zeros(self.T)

		self.lnpi = np.log(pi)
		#print(pi.shape)
		vecEvidence = np.vectorize(squint.lnevidence)

		for t in range(0,self.T):
			lw = self.lnpi + vecEvidence(self.R[t], self.V[t])
			w = np.exp(lw - max(lw))
			w = w / sum(w)	
			 
			self.l_alg[t] = np.dot(w,losses[t])					# instantaneous loss of Squint
			r = self.l_alg[t] - losses[t]    					# vector of instantaneous regrets
			self.R[t] = self.R[t-1] + r    						# update cumulative regret
			self.V[t] = self.V[t-1] + np.square(r)  			# update cumulative variance

def main():
	## Setup
	K = 10;   # number of experts
	T = 100000; # number of rounds

	# prior distribution on experts (drawn uniformly at random from simplex)
	prior = np.sort([np.random.uniform(0,1) for i in range(0,K+1)])
	prior[0] = 0
	prior[-1] = 1

	pi = np.diff(prior)
	print(pi)

	# loss rates of the experts
	rates = np.random.rand(1, K)  # random rates (typically easy data)
	#rates = np.ones((1,K)) # equal uniform rates (worst-case type data)

	losses = np.random.rand(T, K) <= 0.5

	## Run Squint
	s = Squint(pi,losses)

	Losses = np.cumsum(losses, axis = 0)
	L_alg = np.cumsum(s.l_alg)
	L_star = Losses.min(axis = 1)
	print("algorithm losses:", L_alg[-1])
	print("L_star: ", L_star[-1] )
	print("Absolute regret: ", L_alg[-1]-L_star[-1])
	print("Relative regret: ", (L_alg[-1]-L_star[-1])/L_star[-1])

	R_cum = L_alg - L_star

	plt.plot(range(0,T), R_cum);
	plt.title('Cumulative regret');
	# plt.legend(['alg', lg], 'location', 'NorthWest');
	plt.xlabel('T');
	plt.show()



main()