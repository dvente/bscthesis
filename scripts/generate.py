#!/usr/bin/python3
import numpy as np
import random as rand
import sys

#python generate.py <number of cases> <file name>



data = []
for m in range(0,int(sys.argv[1])):
	vec = np.random.normal(0,1,10).tolist()
	if sum(map(lambda x: x**2, vec)) > 9.34:
		vec.append(1)
	else:
		vec.append(0)

	data.append(vec)

with open("../generated/"+ sys.argv[2],'w') as file:
	file.write(str(sys.argv[1])+"\n")
	for i in data:
		for j in i:
			file.write(str(j)+" ")
		file.write("\n")
	