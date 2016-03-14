import numpy as np
import sys
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

def readData(file):
	bin = []
	err = []
	f = open(file,'r')
	for line in f:
		line = line.rstrip()
		array = line.split(" ")
		bin.append(int(array[0]))
		err.append(float(array[-1]))

	return (bin,err)

bin, err = readData("../generated/" + sys.argv[1])


plt.plot(bin, err)
# plt.xlabel('Smarts')
# plt.ylabel('Probability')
# plt.title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')

# Tweak spacing to prevent clipping of ylabel
plt.subplots_adjust(left=0.15)
plt.show()