import multiprocessing
import subprocess
import sys
import bisect as bc
import numpy as np
import matplotlib.pyplot as plt
import argparse

algoDict = {'a' : "AdaBoost", 'n': "NH-Boost.DT", 's':"SquintBoost", 't': "TimSquintBoost"}
dataDict = {'a' : "Ada", 'n': "NH", 's':"SQ", 't': "SQ"}
parser = argparse.ArgumentParser(description='Test an algorithm and plot error as function of number of iterations')
parser.add_argument("--results", default = "../generated/data.dat", help = "location to store the calculated data for later use" )
parser.add_argument("-n", "--notify", help="notify phone when done", action="store_true")
parser.add_argument("-r", "--range", type = int, default = 100, help="Range to test algorithm on")
parser.add_argument("-p", "--plot", default="plot", help="plot the rendered data and store the plot in PLOT")
parser.add_argument("-s", "--show", help="show the plot after it is rendered", action="store_true")
parser.add_argument("-a",'--algorithm', choices=['a','n', 's', 't'], default = 'n', help="Algorithm to render: a(AdaBoost) n(NH-Boos.DT), s(SquintBoost)")
parser.add_argument("-c", "--clean", action = "store_true", help="generate new test and training data before running")
args = parser.parse_args()
trainData = "../generated/" + dataDict[args.algorithm] + "Train.dat"
testData = "../generated/" + dataDict[args.algorithm] + "Test.dat"

def work(N):
	#print('python3.5 ' + args.algorithm + '.py ' + str(N) + ' ' + '--trainData ' + trainData + ' --testData ' + testData)
	#print('python3.5 ' + algoDict[args.algorithm] + '.py ' + str(N) + ' ' + '--trainData ' + trainData + ' --testData ' + testData + ' --log ' + args.results)
	return subprocess.check_output(['python3.5 ' + algoDict[args.algorithm] + '.py ' + str(N) + ' ' + '--trainData ' + trainData + ' --testData ' + testData + ' --log ' + args.results], shell=True).decode("utf-8")

#python render.py -a a -r 500 --results ../generated/ADSVM.dat && python render.py -a n -r 500 --results ../generated/nhtest.dat && python render.py -a s -r 500 --results ../generated/sqtest.dat

#def proces(output, target):
	# for item in output:
	# 	data = item.rstrip()
	# 	data = data.split(" ")
	# 	try:
	# 		data[0] = int(data[0])
	# 	except ValueError:
	# 		print("valerr,",data[0])
	# 	data[1] = float(data[1])
	# 	data[2] = float(data[2])
	# 	data[3] = float(data[3])
	# 	bc.insort(target, data)



if __name__ == '__main__':
	f = open(args.results,'w')
	f.close()
	if(args.clean):
		with open(trainData, 'r') as f:
			first_line = f.readline()
		
		N = int(first_line.rstrip())
		
		with open(testData, 'r') as f:
			first_line = f.readline()

		M = int(first_line.rstrip())
		subprocess.call(["python3.5 generate.py " + str(N) + " " + trainData + " -l 1" ], shell=True)
		subprocess.call(["python3.5 generate.py " + str(M) + " " + testData+ " -l 1" ], shell=True)

	pool = multiprocessing.Pool(None)
	tasks = range(1,args.range)
	results = []
	r = pool.map_async(work, tasks)
	r.wait() # Wait on the results
	# if(args.plot or args.show):
	# 	bins = np.array([item[0] for item in results])
	# 	err = np.array([item[1] for item in results])
	# 	incl = np.array([item[2] for item in results])
	# 	zero = np.array([item[3] for item in results])
		
	# 	with open(trainData, 'r') as f:
	# 		first_line = f.readline()
		
	# 	N = int(first_line.rstrip())
		
	# 	with open(testData, 'r') as f:
	# 		first_line = f.readline()

	# 	M = int(first_line.rstrip())

	# 	stumpAns = subprocess.check_output(['python3.5 tree.py -n 1 ' + ' --testData ' +  testData + ' --trainData ' + trainData], shell=True).decode("utf-8")
	# 	stumpAns = stumpAns.rstrip()
	# 	stumpAns = stumpAns.split(" ")
	# 	treeAns = subprocess.check_output(['python3.5 tree.py -n 500 ' + ' --testData ' +  testData + ' --trainData ' + trainData], shell=True).decode("utf-8")
	# 	treeAns = treeAns.rstrip()
	# 	treeAns = treeAns.split(" ")

	# 	plt.axhline(y=float(stumpAns[1]), xmin = 0, xmax = 1, hold = None, ls = 'dashed')
	# 	plt.axhline(y=float(treeAns[1]), xmin = 0, xmax = 1, hold = None, ls = 'dashed')
	# 	plt.text(len(err)//2,float(treeAns[1])+0.001,'Decision tree with ' + treeAns[0] + ' nodes',color='b')
	# 	plt.text(len(err)//4,float(stumpAns[1])+0.001,'Decision stump',color='b')
	# 	#print(M//2)
		
	# 	with open(args.results,'w') as file:
	# 		for i in range(0,len(bins)):
	# 			file.write(str(bins[i])+" " + str(err[i]) +  " " +  str(incl[i]) + " " + str(zero[i]) + "\n")

	# 	plt.xlabel('T')
	# 	plt.ylabel('Generalization Error')
	# 	plt.title("Plot of " + algoDict[args.algorithm] + " error: $N="+ str(N) + "$, $M=" + str(M) +"$")
	# 	plt.subplots_adjust(left = 0.15)
		
	# 	plt.plot(bins, err, color='r', label="Generalization error (%)")
	# 	plt.plot(bins, incl, color='b', label="Inconclusive tests (%)")
	# 	if(args.algorithm == 'n'):
	# 		plt.plot(bins, zero, color='g', label="Number of zero weights (%)")
	# 	plt.legend()
	# 	plt.axis([0,args.range,0,0.5])
	# 	plt.grid()
	# 	if(args.plot):
	# 		plt.savefig("../generated/" +  args.plot +".png")
	# 	if(args.notify):
	# 		subprocess.Popen(["/bin/bash", "-i", "-c", "push \"Python script done\" \"Your algorithm is done rendering\""])

	# 	if(args.show):
	# 		plt.show()
	# else:
	# 	print(results)
	# 	if(args.notify):
	# 		subprocess.Popen(["/bin/bash", "-i", "-c", "push \"Python script done\" \"Your algorithm is done rendering\""])
