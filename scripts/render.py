import multiprocessing
import subprocess
import sys
import bisect as bc
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Test an algorithm and plot error as function of number of iterations')
parser.add_argument('--trainData', default = "../generated/DTtrain.dat",help="location of the training data")
parser.add_argument("--testData", default = "../generated/DTtest.dat", help = "location of the test data" )
parser.add_argument("--results", default = "../generated/DTdata.dat", help = "location to store the calculated data for later use" )
parser.add_argument("-n", "--notify", help="notify phone when done", action="store_true")
parser.add_argument("-r", "--range", type = int, default = 100, help="Range to test algorithm on")
parser.add_argument("-p", "--plot", default="plot", help="plot the rendered data and store the plot in PLOT")
parser.add_argument("-s", "--show", help="show the plot after it is rendered", action="store_true")
parser.add_argument("-a",'--algorithm', default = "AdaBoost",help="Algorithm to render errors of")
parser.add_argument("-c", "--clean", action = "store_true", help="generate new test and training data before running")
args = parser.parse_args()

def work(N):
	#print('python3.4 ' + args.algorithm + '.py ' + str(N) + ' ' + '--trainData ' + args.trainData + ' --testData ' + args.testData)
	return subprocess.check_output(['python3.4 ' + args.algorithm + '.py ' + str(N) + ' ' + '--trainData ' + args.trainData + ' --testData ' + args.testData], shell=True).decode("utf-8")

def proces(output, target):
	for item in output:
		data = item.rstrip()
		data = data.split(" ")
		try:
			data[0] = int(data[0])
		except ValueError:
			print("valerr,",data[0])
		data[1] = float(data[1])
		data[2] = float(data[2])
		data[3] = float(data[3])
		bc.insort(target, data)

if __name__ == '__main__':
	if(args.clean):
		with open(args.trainData, 'r') as f:
			first_line = f.readline()
		
		N = int(first_line.rstrip())
		
		with open(args.testData, 'r') as f:
			first_line = f.readline()

		M = int(first_line.rstrip())
		subprocess.call(["python3.4 generate.py " + str(N) + " " + args.trainData + " -l 1" ], shell=True)
		subprocess.call(["python3.4 generate.py " + str(M) + " " + args.testData+ " -l 1" ], shell=True)

	pool = multiprocessing.Pool(None)
	tasks = range(1,args.range)
	results = []
	r = pool.map_async(work, tasks, callback=lambda x:proces(x, results))
	r.wait() # Wait on the results
	if(args.plot or args.show):
		bins = np.array([item[0] for item in results])
		err = np.array([item[1] for item in results])
		incl = np.array([item[2] for item in results])
		zero = np.array([item[3] for item in results])
		
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

		plt.axhline(y=float(stumpAns[1]), xmin = 0, xmax = 1, hold = None, ls = 'dashed')
		plt.axhline(y=float(treeAns[1]), xmin = 0, xmax = 1, hold = None, ls = 'dashed')
		plt.text(len(err)//2,float(treeAns[1])+0.001,'Decision tree with ' + treeAns[0] + ' nodes',color='b')
		plt.text(len(err)//2,float(stumpAns[1])+0.001,'Decision stump',color='b')
		#print(M//2)
		
		with open(args.results,'w') as file:
			for i in range(0,len(bins)):
				file.write(str(bins[i])+" " + str(err[i]) + "\n")

		plt.xlabel('T')
		plt.ylabel('Generalization Error')
		plt.title("Plot of " + str(args.algorithm)+ " error: $N="+ str(N) + "$, $M=" + str(M) +"$")
		plt.subplots_adjust(left = 0.15)
		
		plt.plot(bins, err, color='r')
		plt.plot(bins, incl, color='b')
		plt.plot(bins, zero, color='g')
		plt.grid()
		if(args.plot):
			plt.savefig("../generated/" +  args.plot +".png")
		if(args.notify):
			subprocess.call(['push "Python script" "Your script is done rendering"'], shell=True)
		if(args.show):
			plt.show()
	else:
		print(results)
		if(args.notify):
			subprocess.call(['push "Python script" "Your script is done rendering"'], shell=True)
