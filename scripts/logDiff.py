import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt

def logDiff(a,b):
	if(np.any(a<b)):
		raise DomainError("a<b undefined in logDiff")
	x = np.log(a)
	y = np.log(b)
	v = np.log(max(a,b))
	return np.log(np.exp(x-v)-np.exp(y-v))+v

# for i in range(0,10):
# 	a = abs(np.random.normal())
# 	b = abs(np.random.normal())
# 	print((max(a,b)-min(a,b))-np.exp(logDiff(max(a,b),min(a,b))))


# a = np.arange(-100,100)
# b = np.square([min(0,x+1) for x in a ])
# a = np.square([min(0,x-1) for x in a ])
# plt.plot(a, color='r')
# plt.plot(b)
# plt.show()
