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

bin, err = readData(sys.argv[1])


plt.plot(bin, err)
plt.axhline(y=float(" 0.4602"), xmin=0, xmax=1, hold=None, ls='dashed')
plt.savefig(sys.argv[1][:-4]+".png")
plt.subplots_adjust(left=0.15)
plt.show()