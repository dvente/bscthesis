# Copyright 2016 Daniel Vente
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of version 3 of the GNU General Public License as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import multiprocessing
import subprocess
import argparse

parser = argparse.ArgumentParser(
            description="""This program will test an algorithm and plot error 
                        as function of number of iterations. Multi-core 
                        processing is used for speed up, this is quite intensive!
                        Author: Daniel Vente <danvente@gmail.com> June 2016""",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--trainData", help="Location of the training data.")
parser.add_argument("--testData",  help = "Location of the test data.")
parser.add_argument("--results", default = "../generated/data.dat", 
                    help = "Location to store the calculated data." )
parser.add_argument("-r", "--range", type = int, default = 100, 
                    help="Upper bound on range to test algorithm on")
parser.add_argument("-p", "--plot", 
                    help="Plot the rendered data and store the plot in PLOT.")
parser.add_argument("-s", "--show", 
                     help="Plot the rendered data, store it in SHOW and show it.")
parser.add_argument("-a","--algorithm", choices=["a", "n", "s"], 
        default = "n", help="Algorithm to render: a(AdaBoost) n(NH-BoostDT), s(SquintBoost)")
args = parser.parse_args()

#A slightly more sophisticated default for the train and test data. 
algoDict = {"a" : "AdaBoost", "n": "NH-BoostDT", "s":"SquintBoost"}
dataDict = {"a" : "Ada", "n": "NH", "s":"SQ"}
if args.trainData is None:
    trainData = "../data/" + dataDict[args.algorithm] + "Train.dat"
if args.testData is None:
    testData = "../data/" + dataDict[args.algorithm] + "Test.dat"

def work(N):
    return subprocess.check_output([
            "python3 " + algoDict[args.algorithm] + ".py " 
            + str(N) + " " 
            + "--trainData " + trainData 
            + " --testData " + testData 
            + " --log " + args.results], shell=True).decode("utf-8")

if __name__ == "__main__":
    with open(args.results, "w"):
        pass
    pool = multiprocessing.Pool(None)
    tasks = range(args.range)
    results = []
    r = pool.map_async(work, tasks)
    r.wait() # Wait on the results

    #sort the file
    p1 = subprocess.Popen(['cat', args.results], stdout=subprocess.PIPE)
    p2 = subprocess.Popen(['sort', '-n'], stdin=p1.stdout, stdout=subprocess.PIPE)
    with open(args.results, "w") as f:
        f.write(p2.communicate()[0].decode('ascii'))

    if args.plot is not None:
       subprocess.call([
            "python3 plot.py " 
            + args.results + " " 
            + args.plot 
            + " -a " + args.algorithm 
            + " --trainData " + trainData 
            + " --testData " + testData ], shell=True)
    elif args.show is not None:
        subprocess.call([
            "python3 plot.py " 
            + args.results + " " 
            + args.show 
            + " -a " + args.algorithm 
            + " --trainData " + trainData 
            + " --testData " + testData 
            + " -s "], shell=True)