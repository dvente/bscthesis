import numpy as np
import subprocess
import argparse

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
        description=""" This program plots the data from the files generated 
                        by the other scripts. The format must be as follows:
                        Trails error inconclusive zeros
                        Author: Daniel Vente <danvente@gmail.com> June 2016
                        """,
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
        "-a","--algorithm", choices=["a", "n", "s"], default = "a", 
        help="""Algorithm used to generate the data: 
                (a:AdaBoost, n:NH-Boos.DT, s:SquintBoost)""")
parser.add_argument(
        "-s", "--show", action="store_true",
        help="Show the plot after storing it.")
parser.add_argument("--a9a", action="store_true",
        help="Shorthand to reference trees against a9a.")
parser.add_argument("--trainData", help="""Location of the training data for 
                                            tree reference""")
parser.add_argument("--testData", help="""Location of the test data for 
                                            tree reference""")
parser.add_argument("file", help="Location of the data to plot.")
parser.add_argument("filename", help="Location to store the plot.")
args = parser.parse_args()

algoDict = {"a" : "Ada", "n": "NH", "s":"SQ"}
if args.a9a is False:
    testData = "../data/" + algoDict[args.algorithm] + "Test.dat"
    trainData = "../data/" + algoDict[args.algorithm] + "Train.dat"
else:
    testData = "../data/a9a"
    trainData = "../data.a9a.t"

if args.trainData is not None:
    testData = args.trainData
if args.testData  is not None:
    trainData = args.trainData

bins = []
err = []
incl = []
zero = []
f = open(args.file,"r")
for line in f:
    line = line.rstrip()
    array = line.split(" ")
    if array[0] == '':
        continue
    bins.append(int(array[0]))
    err.append(float(array[1]))
    if args.algorithm != "AdaBoost":
        incl.append(float(array[2]))
        zero.append(float(array[3]))

#Inconclusive tests on odd iterations are always 0 so strip them
if args.algorithm != "AdaBoost": 
    inclBins = bins[3::2]
    inclStrip = incl[3::2]

#Get the tree"s performance on the relevant data for referance

stumpAns = subprocess.check_output(
    ["python3.5 tree.py -n 1 " 
    + " --testData " +  testData 
    + " --trainData " + trainData],
    shell=True).decode("utf-8")
stumpAns = stumpAns.rstrip()
stumpAns = stumpAns.split(" ")

treeAns = subprocess.check_output(
    ["python3.5 tree.py -n 500 " 
    + " --testData " +  testData 
    + " --trainData " + trainData],
    shell=True).decode("utf-8")
treeAns = treeAns.rstrip()
treeAns = treeAns.split(" ")

#Plot peformance of the trees.
plt.axhline(y=float(stumpAns[1]), xmin=0, xmax=1, 
            hold=None, ls="dashed", color="black")
plt.axhline(y=float(treeAns[1]), xmin=0, xmax=1, 
            hold=None, ls="dashed", color="black")

plt.text(len(err)//2, float(treeAns[1])+0.001, 
        "Decision tree with " + treeAns[0] + " nodes", color="black")
plt.text(len(err)//7, float(stumpAns[1])+0.001, 
        "Decision stump", color="black")

#Plot everything else.
plt.xlabel("T")
plt.ylabel("Generalization Error")
plt.subplots_adjust(left = 0.15)

plt.plot(bins, err, color="r", label="Generalization error (%)")
if args.algorithm != "AdaBoost":
    plt.plot(inclBins, inclStrip, color="b", label="Inconclusive tests (%)")
if args.algorithm == "NH-Boost.DT":
    plt.plot(bins, zero, color="g", label="Number of zero weights (%)")
if args.algorithm != "AdaBoost":
    plt.legend()

plt.grid()
plt.axis([0,max(bins)+1,0,0.5])
plt.xticks(np.arange(0, max(bins)+1, (max(bins)+1)/10))
plt.yticks(np.arange(0, 0.5, 0.05))
plt.savefig("../generated/" + args.filename + ".png")

if(args.show):
    plt.show()