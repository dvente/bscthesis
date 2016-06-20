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


import numpy as np
import argparse
import squint

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_svmlight_file

def isDist(array):
    return (np.all(array>=0) and np.isclose(sum(array),1.0))

vecEvidence = np.vectorize(squint.lnevidence)

class SquintBoostClassifier:
    def __init__(self, T):
        self.T = T + 1
        self.weakLearn = np.array(
                [DecisionTreeClassifier(max_depth = 1) 
                for i in range(0,self.T)])

    def fit(self, X, y):

        if len(np.unique(y)) is not 2:
            raise ValueError(
                    """"Too many labels detected, this implementation only 
                        handles binary classification 
                    """)
        else:
        #The algorithm depends on labels being -1 and 1 so make a dummy list
        # with the right labels. 
            self.maxLabel = max(y)
            self.minLabel = min(y)
            Y = y
            Y[Y == self.maxLabel] = 1
            Y[Y == self.minLabel] = -1

        R = np.zeros(X.shape[0])
        V = np.zeros(X.shape[0])

        for t in range(1,self.T):
            lw = vecEvidence(R, V)
            w = np.exp(lw)
            p = w / sum(w) 

            self.weakLearn[t].fit(X, y, sample_weight=p)
            margins = self.weakLearn[t].predict(X)*y
            gamma = np.sum(p*self.weakLearn[t].predict(X)*y)
            r = 0.5*(gamma-margins) 
            R = R + r
            V = V + np.square(r)



    def predict(self, X):
        ans = np.sign(sum([self.weakLearn[t].predict(X) 
                            for t in range(1,self.T)]))
        if ans == [1]:
            return self.maxLabel
        elif ans == [-1]:
            return self.minLabel
        else:
            return float('Inf')

parser = argparse.ArgumentParser(
            description="""This program uses Squint-Boost to (binary) classify 
                        the test data based upon the provided training data and 
                        reports the number of iterations and the error rate in
                        the following format:

                        Trails error inconclusive 0.0.
                        
                        All outputs except the trails are percentages. 
                        The trailing 0 is added for consistency with 
                        the other two algorithms used when plotting. 
                        Author: Daniel Vente <danvente@gmail.com> June 2016""",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("trails", type = int, 
                    help="Number of trails with which to train Squint-Boost.")
parser.add_argument("--trainData", default = "../data/SQTrain.dat", 
                    help="Location of the training data.")
parser.add_argument("--testData", default = "../data/SQTest.dat", 
                    help = "Location of the test data.")
parser.add_argument("--log", help="Store answer to file instead of printing" )
parser.add_argument("-v", "--verbose", action="store_true", 
                    help="Increase verbosity to improve readability (for humans)" )
args = parser.parse_args()

s = SquintBoostClassifier(args.trails)

X, y = load_svmlight_file(args.trainData)
testX, testY = load_svmlight_file(args.testData)

if X.shape[0] != y.shape[0]:
    raise ValueError("X has %d examples instead of %d like y", 
                        X.shape[0], y.shape[0])

if testX.shape[0] != testY.shape[0]:
    raise ValueError("testX has %d examples instead of %d like testY", 
                        testX.shape[0], testY.shape[0])

s.fit(X,y)
ansArr = np.array([s.predict(testX[i]) for i in range(testX.shape[0])])
boostErr, incl = 0, 0

for t in range(testX.shape[0]):
    if(ansArr[t] == float('Inf')):
        incl += 1
        boostErr += 0.5
    elif(ansArr[t] != testY[t]):
        boostErr += 1  

if args.verbose is False:
    output = "{0} {1} {2} 0.0\n".format(
            args.trails,
            boostErr/testX.shape[0],
            incl/testX.shape[0])
else:
    output = "Trails: {0}\nError(%): {1}\nInconclusive(%): {2}\nZeros(%): 0.0\n".format(
            args.trails,
            boostErr/testX.shape[0],
            incl/testX.shape[0])

if args.log is not None:
    with open(args.log,"a") as file:
        file.write(output)
else:
    print(output)