import numpy as np
import random as rand
import sys

data = []
for m in range(0,int(sys.argv[1])):
	vec = np.random.normal(0,1,10).tolist()
	if sum(map(lambda x: x**2, vec)) > 9.34:
		vec.append(1)
	else:
		vec.append(-1)

	data.append(vec)

with open(sys.argv[2],'w') as file:
	for i in data:
		for j in i:
			file.write(str(j)+" ")
		file.write("\n")
	