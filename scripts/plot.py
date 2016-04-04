import numpy as np
import argparse
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

parser = argparse.ArgumentParser(description='Plot data from a two column .csv or .log file')
parser.add_argument('file', help='location of the data to plot')
args = parser.parse_args()

bin, err = readData(args.file)

plt.plot(bin, err)
plt.savefig(args.file[:-4]+".png")
plt.subplots_adjust(left=0.15)
plt.show()