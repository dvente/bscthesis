import multiprocessing
import subprocess
import sys
import bisect as bc
import numpy as np
import matplotlib.pyplot as plt

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
		
		plt.xlabel('T')
		plt.ylabel('Generalization Error')
		plt.title("Plot of AdaBoost error: $N="+ str(N) + "$, $M=" + str(M) +"$")
		plt.subplots_adjust(left=0.15)
		
		plt.plot(bins, err)
		plt.savefig("../generated/" +  sys.argv[2]+".png")
		subprocess.call(['pushover --api-token aqz4SVrrb5a67EwnytQvmfnrYUnifw --user-key uUNPbABuEqPWvR5Y9agZeB59ZiMkqo "Your script is done"'], shell=True)
		plt.show()
	else:
		print(results)
		subprocess.call(['pushover --api-token aqz4SVrrb5a67EwnytQvmfnrYUnifw --user-key uUNPbABuEqPWvR5Y9agZeB59ZiMkqo "Your script is done"'], shell=True)
	