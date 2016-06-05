import mpmath as mp
import numpy as np
import math
import random

def magnitude(x):
	if(x == 0):
		return 0
	else:
		return int(np.log10(np.abs(x)))

def mutate(function, start, tolerence):
	for t in range(1,100):
		org = function(start,t)
		for i in range(0,100):
			mut = np.random.normal() * mp.power(10,magnitude(start)-2)
			new = function(start+mut,t)
			dx = abs(abs(org - new)/mut)
			if(dx > tolerence and tolerence is not 0):
				raise ValueError("Numerical inconsistency in " + str(function.__name__) + " at (" + str(start) + "," +str(t)+") + " + str(mut) + "with tolerence " + str(tolerence))

def setS(x,t):
	# math.exp(np.square()/(3*t))
	# math.exp(np.square(min(0,x+1))/(3*t))
	return math.exp(np.square(min(0,x-1))/(3*t))-math.exp(np.square(min(0,x+1))/(3*t))

# def test(x):
# 	return mp.exp(-x)-mp.exp(2*x*10)
for i in [np.power(1,-j) for j in range(0,20)]:
	mutate(setS, np.random.normal(scale=i), magnitude(i))
print("\"Numercially stable\"")