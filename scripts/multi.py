import multiprocessing
import subprocess

def work(N):
    return subprocess.call(['python3.4 AdaBoost.py ' + str(N)], shell=True)

if __name__ == '__main__':
    pool = multiprocessing.Pool(None)
    tasks = range(1,100)
    results = []
    r = pool.map_async(work, tasks, callback=results.append)
    r.wait() # Wait on the results
    #print(results)
    subprocess.call(['pushover --api-token aqz4SVrrb5a67EwnytQvmfnrYUnifw --user-key uUNPbABuEqPWvR5Y9agZeB59ZiMkqo "Your script is done"'], shell=True)