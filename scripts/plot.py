import numpy as np
import argparse
# import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import subprocess

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

algoDict = {"AdaBoost" : "Ada", "NH-Boost.DT" : "NH", "SquintBoost": "SQ"}

parser = argparse.ArgumentParser(description='Plot data from a four column .csv or .log file')
parser.add_argument('algo', help="Name of the algorithm")
parser.add_argument('file', help='location of the data to plot')
parser.add_argument('title', help="Location to store the plot")
parser.add_argument('--svm', action="store_true")
parser.add_argument('-s', '--show', action="store_true")
args = parser.parse_args()

if not args.svm:
	testData = "../data/" + algoDict[args.algo] + "Test.dat"
	trainData = "../data/" + algoDict[args.algo] + "Train.dat"
	dataSet = "simulated "
else:
	dataSet = "a9a "

bins = []
err = []
incl = []
zero = []
f = open(args.file,'r')
for line in f:
	line = line.rstrip()
	array = line.split(" ")
	bins.append(int(array[0]))
	err.append(float(array[1]))
	if args.algo != "AdaBoost":
		incl.append(float(array[2]))
		zero.append(float(array[3]))



N = 32561
M = 16281

if args.algo != "AdaBoost":
	inclBins = bins[3::2]
	inclStrip = incl[3::2]
	#zeroStrip = zero[3::2]

if args.svm:
	stumpAns = subprocess.check_output(['python3.5 tree.py -n 1 --svm'], shell=True).decode("utf-8")
	stumpAns = stumpAns.rstrip()
	stumpAns = stumpAns.split(" ")
	treeAns = subprocess.check_output(['python3.5 tree.py -n 500 --svm'], shell=True).decode("utf-8")
	treeAns = treeAns.rstrip()
	treeAns = treeAns.split(" ")
else:
	stumpAns = subprocess.check_output(['python3.5 tree.py -n 1 ' + ' --testData ' +  testData + ' --trainData ' + trainData], shell=True).decode("utf-8")
	stumpAns = stumpAns.rstrip()
	stumpAns = stumpAns.split(" ")
	treeAns = subprocess.check_output(['python3.5 tree.py -n 500 ' + ' --testData ' +  testData + ' --trainData ' + trainData], shell=True).decode("utf-8")
	treeAns = treeAns.rstrip()
	treeAns = treeAns.split(" ")

plt.axhline(y=float(stumpAns[1]), xmin = 0, xmax = 1, hold = None, ls = 'dashed', color='black')
plt.axhline(y=float(treeAns[1]), xmin = 0, xmax = 1, hold = None, ls = 'dashed', color ='black')
plt.text(len(err)//2,float(treeAns[1])+0.001,'Decision tree with ' + treeAns[0] + ' nodes',color='black')
plt.text(len(err)//7,float(stumpAns[1])+0.001,'Decision stump',color='black')


plt.xlabel('T')
plt.ylabel('Generalization Error')
plt.title("Plot of " + args.algo + " error with the " + dataSet + "data set: $N="+ str(N) + "$, $M=" + str(M) +"$")
plt.subplots_adjust(left = 0.15)

plt.plot(bins, err, color='r', label="Generalization error (%)")
if args.algo != "AdaBoost":
	plt.plot(inclBins, inclStrip, color='b', label="Inconclusive tests (%)")
if args.algo == "NH-Boost.DT":
	plt.plot(bins, zero, color='g', label="Number of zero weights (%)")
if args.algo != "AdaBoost":
	plt.legend()
plt.grid()
plt.axis([0,len(bins),0,0.5])
plt.xticks(np.arange(0, 500, 50))
plt.yticks(np.arange(0, 0.5, 0.05))
plt.savefig("../generated/" + args.title + ".png")

if(args.show):
	plt.show()