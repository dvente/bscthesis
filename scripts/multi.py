import multiprocessing
import subprocess
import sys
import bisect as bc
import numpy as np
import matplotlib.pyplot as plt
import argparse

#python multi.py <test trails> <plot file name>

def work(N):
    return subprocess.check_output(['python3.4 AdaBoost.py ' + str(N) + ' ' + str(sys.argv[1])], shell=True).decode("utf-8")

def proces(output, target):
	for item in output:
		data = item.rstrip()
		data = data.split(" ")
		data[0] = int(data[0])
		data[1] = float(data[1])
		bc.insort(target, data)


if __name__ == '__main__':
	pool = multiprocessing.Pool(None)
	tasks = range(1,100)
	results = []
	r = pool.map_async(work, tasks, callback=lambda x:proces(x, results))
	r.wait() # Wait on the results
	if(len(sys.argv)>1):
		bins = np.array([item[0] for item in results])
		err = np.array([item[1] for item in results])
		
		with open('../generated/train.dat', 'r') as f:
			first_line = f.readline()
		
		N = int(first_line.rstrip())
		M = int(sys.argv[1])

		stumpAns = subprocess.check_output(['python3.4 tree.py 1 ' + str(sys.argv[1])], shell=True).decode("utf-8")
		stumpAns = stumpAns.rstrip()
		stumpAns = stumpAns.split(" ")
		treeAns = subprocess.check_output(['python3.4 tree.py ' + str(500) + ' ' + str(sys.argv[1])], shell=True).decode("utf-8")
		treeAns = treeAns.rstrip()
		treeAns = treeAns.split(" ")

		plt.axhline(y=float(stumpAns[1]), xmin=0, xmax=1, hold=None, ls='dashed')
		plt.axhline(y=float(treeAns[1]), xmin=0, xmax=1, hold=None, ls='dashed')
		#plt.axvline(y=float(treeAns[1]))
		plt.text(len(err)//2,float(treeAns[1])+0.001,'Decision tree with ' + treeAns[0] + ' nodes',color='b')
		plt.text(len(err)//2,float(stumpAns[1])+0.001,'Decision stump',color='b')
		#print(M//2)
		
		with open("../generated/data.dat",'w') as file:
			for i in range(0,len(bins)):
				file.write(str(bins[i])+" " + str(err[i]) + "\n")

		plt.xlabel('T')
		plt.ylabel('Generalization Error')
		plt.title("Plot of AdaBoost error: $N="+ str(N) + "$, $M=" + str(M) +"$")
		plt.subplots_adjust(left=0.15)
		
		plt.plot(bins, err, color='r')
		plt.savefig("../generated/" +  sys.argv[2]+".png")
		subprocess.call(['pushover --api-token aqz4SVrrb5a67EwnytQvmfnrYUnifw --user-key uUNPbABuEqPWvR5Y9agZeB59ZiMkqo "Your script is done"'], shell=True)
		#plt.show()
	else:
		print(results)
		subprocess.call(['pushover --api-token aqz4SVrrb5a67EwnytQvmfnrYUnifw --user-key uUNPbABuEqPWvR5Y9agZeB59ZiMkqo "Your script is done"'], shell=True)
	