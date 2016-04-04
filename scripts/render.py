import multiprocessing
import subprocess
import sys
import bisect as bc
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Test an algorithm and plot error as function of number of iterations')
parser.add_argument('--trainData', default = "../generated/train.dat",help="location of the training data")
parser.add_argument("--testData", default = "../generated/test.dat", help = "location of the test data" )
parser.add_argument("--results", default = "../generated/data.dat", help = "location to store the calculated data for later use" )
parser.add_argument("-n", "--notify", help="notify phone when done", action="store_true")
parser.add_argument("-r", "--range", type = int, default = 50, help="Range to test algorithm on")
parser.add_argument("-p", "--plot", help="plot the rendered data and store the plot in PLOT")
parser.add_argument("-s", "--show", help="show the plot after it is rendered", action="store_true")
args = parser.parse_args()

def work(N):
    return subprocess.check_output(['python3.4 AdaBoost.py ' + str(N) + ' ' + '--trainData ' + args.trainData], shell=True).decode("utf-8")

def proces(output, target):
	for item in output:
		data = item.rstrip()
		data = data.split(" ")
		data[0] = int(data[0])
		data[1] = float(data[1])
		bc.insort(target, data)


if __name__ == '__main__':
	pool = multiprocessing.Pool(None)
	tasks = range(1,args.range)
	results = []
	r = pool.map_async(work, tasks, callback=lambda x:proces(x, results))
	r.wait() # Wait on the results
	if(args.plot):
		bins = np.array([item[0] for item in results])
		err = np.array([item[1] for item in results])
		
		with open(args.trainData, 'r') as f:
			first_line = f.readline()
		
		N = int(first_line.rstrip())
		
		with open(args.testData, 'r') as f:
			first_line = f.readline()

		M = int(first_line.rstrip())

		stumpAns = subprocess.check_output(['python3.4 tree.py -n 1 ' + ' --testData ' +  args.testData + ' --trainData ' + args.trainData], shell=True).decode("utf-8")
		stumpAns = stumpAns.rstrip()
		stumpAns = stumpAns.split(" ")
		treeAns = subprocess.check_output(['python3.4 tree.py -n 500 ' + ' --testData ' +  args.testData + ' --trainData ' + args.trainData], shell=True).decode("utf-8")
		treeAns = treeAns.rstrip()
		treeAns = treeAns.split(" ")

		plt.axhline(y=float(stumpAns[1]), xmin=0, xmax=1, hold=None, ls='dashed')
		plt.axhline(y=float(treeAns[1]), xmin=0, xmax=1, hold=None, ls='dashed')
		#plt.axvline(y=float(treeAns[1]))
		plt.text(len(err)//2,float(treeAns[1])+0.001,'Decision tree with ' + treeAns[0] + ' nodes',color='b')
		plt.text(len(err)//2,float(stumpAns[1])+0.001,'Decision stump',color='b')
		#print(M//2)
		
		with open(args.results,'w') as file:
			for i in range(0,len(bins)):
				file.write(str(bins[i])+" " + str(err[i]) + "\n")

		plt.xlabel('T')
		plt.ylabel('Generalization Error')
		plt.title("Plot of AdaBoost error: $N="+ str(N) + "$, $M=" + str(M) +"$")
		plt.subplots_adjust(left=0.15)
		
		plt.plot(bins, err, color='r')
		plt.savefig("../generated/" +  args.plot +".png")
		if(args.notify):
			subprocess.call(['pushover --api-token aqz4SVrrb5a67EwnytQvmfnrYUnifw --user-key uUNPbABuEqPWvR5Y9agZeB59ZiMkqo "Your script is done"'], shell=True)
		if(args.show):
			plt.show()
	else:
		print(results)
		if(args.notify):
			subprocess.call(['pushover --api-token aqz4SVrrb5a67EwnytQvmfnrYUnifw --user-key uUNPbABuEqPWvR5Y9agZeB59ZiMkqo "Your script is done"'], shell=True)
	