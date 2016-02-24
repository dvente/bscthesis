import hedge
import string
import random as rand
import numpy as np
import multiprocessing as mult

betaSet = []
lossSet = []

def learn(beta):
	h = hedge.hedgeL(100, hedge.uniformDist(10),hedge.kPredict,beta)
	h.train()
	return (h.beta, h.cumLoss)

if __name__ == '__main__':
	pool = mult.Pool(processes=4)
	data = pool.map(learn,np.arange(0.1,0.5,0.01))
	beta, loss = zip(*data)
	print beta[loss.index(min(loss))]


# minans = 1000
# minw = []
# minbeta = 0
# for beta in np.arange(0.0001,0.5,0.000001):
# 	h = hedge.hedgeL(100, hedge.uniformDist(10),hedge.kPredict,beta)
# 	h.train()
# 	if h.cumLoss < minans:
# 		minans = h.cumLoss
# 		minw = h.w
# 		minbeta = beta
# print minans
# h = hedge.hedgeL(10000, hedge.uniformDist(10),hedge.kPredict)
# h.train()
# print h.cumLoss
#16.5024545337