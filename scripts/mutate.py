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
	org = function(start)
	for i in range(0,10):
		mut = np.random.normal() * mp.power(10,magnitude(start)-2)
		#print (mut)
		new = function(start+mut)
		dx = abs(abs(org - new)/mut)
		print(dx)
		if(dx > tolerence):
			raise ValueError("Numerical inconsistency in " + str(function.__name__) + " at " + str(start) + " + " + str(mut) + "with tolerence " + str(tolerence))

vMutate = np.vectorize(mutate)
vMagnitude = np.vectorize(magnitude)
# def test(x):
# 	return mp.exp(-x)-mp.exp(2*x*10)

# mutate(test, 1, 1)
# print("\"Numercially stable\"")