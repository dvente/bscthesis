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

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_svmlight_file


def isDist(array):
    return (np.all(array>=0) and np.isclose(sum(array),1.0))

def neg(s):
  return min(0,s)   

vNeg = np.vectorize(neg)


class NHBoostDTClassifier:
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

        s = np.zeros(X.shape[0])
        self.p = np.zeros(X.shape[0])

        for t in range(1,self.T):
            self.p = (np.exp(np.square(vNeg(s-1))/(3*t))
                -np.exp(np.square(vNeg(s+1))/(3*t)))
            self.p = self.p / sum(self.p)
            if not isDist(self.p):
                raise ValueError("non dist")
            self.weakLearn[t].fit(X, Y, sample_weight = self.p)
            gamma = sum(self.p*Y*self.weakLearn[t].predict(X))/2
            s = s + Y*self.weakLearn[t].predict(X) - gamma

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
            description="""This program uses NH-Boost.DT to (binary) classify the 
                        test data based upon the provided training data and 
                        reports the number of iterations and the error rate in
                        the following format:

                        Trails error inconclusive zeros.
                        
                        All outputs except the trails are percentages.
                        Author: Daniel Vente <danvente@gmail.com> June 2016""",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("trails", type = int, 
                    help="Number of trails with which to train NH-Boost.DT.")
parser.add_argument("--trainData", default = "../data/NHTrain.dat", 
                    help="Location of the training data.")
parser.add_argument("--testData", default = "../data/NHTest.dat", 
                    help = "Location of the test data.")
parser.add_argument("--log", help="Store answer to file instead of printing" )
parser.add_argument("-v", "--verbose", action="store_true", 
                    help="Increase verbosity to improve readability (for humans)" )
args = parser.parse_args()

n = NHBoostDTClassifier(args.trails)

X, y = load_svmlight_file(args.trainData)
testX, testY = load_svmlight_file(args.testData)

if X.shape[0] != y.shape[0]:
    raise ValueError("X has %d examples instead of %d like y", 
                        X.shape[0], y.shape[0])

if testX.shape[0] != testY.shape[0]:
    raise ValueError("testX has %d examples instead of %d like testY", 
                        testX.shape[0], testY.shape[0])

n.fit(X,y)
ansArr = np.array([n.predict(testX[i]) for i in range(testX.shape[0])])
boostErr, incl = 0, 0

for t in range(testX.shape[0]):
    if(ansArr[t] == float('Inf')):
        incl += 1
        boostErr += 0.5
    elif(ansArr[t] != testY[t]):
        boostErr += 1


if args.verbose is False:
    output = "{0} {1} {2} {3}\n".format(
            args.trails,
            boostErr/testX.shape[0],
            incl/testX.shape[0],
            (n.p.shape[0]-np.count_nonzero(n.p))/n.p.shape[0])
else:
    output = "Trails: {0}\nError(%): {1}\nInconclusive(%): {2}\nZeros(%): {3}\n".format(
            args.trails,
            boostErr/testX.shape[0],
            incl/testX.shape[0],
            (n.p.shape[0]-np.count_nonzero(n.p))/n.p.shape[0])

if args.log is not None:
    with open(args.log,"a") as file:
        file.write(output)
else:
    print(output)

