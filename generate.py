import numpy as np
import sys
import pandas

data = np.array([np.random.normal(0,1,11).tolist() for i in range(0, int(sys.argv[1]))])
for row in data:
	if np.sum(map(lambda x: x**2, row[:-2])) > 9.34:
		row[10] = 1
	else:
		row[10] = -1 

D = pandas.DataFrame(data)
D.to_csv("train.dat", index = False)
