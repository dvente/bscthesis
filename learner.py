import hedge
import string
import random as rand
import numpy as np
import multiprocessing as mult

def learn(beta):
	h = hedge.hedgeL(100, hedge.uniformDist(10),hedge.kPredict,beta)
	h.train()
	return h.cumLoss

if __name__ == '__main__':
	pool = mult.Pool(processes=4)
	#for beta in np.arange(0.0001,0.5,0.000001):
	data = pool.map(learn,np.arange(0.0001,0.5,0.000001))
	print min(data)


