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

class AdaBoostClassifier:
    def __init__(self, T):
    #Store only the important things in member variables
        if T < 1:
            raise ValueError("Cannot run less than 1 iteration")

        self.T = T 
        self.beta = np.zeros((self.T,1))
        self.weakLearn = np.array(  
                [DecisionTreeClassifier(max_depth = 1) 
                for i in range(self.T)])
        self.thresh = 0

    def fit(self, X, y, sampleWeight=None):
        
        #Check sampleWeight is in a valid format
        if sampleWeight is None:
            w = np.array([(1/X.shape[0]) for i in range(X.shape[0]) ])
        elif sampleWeight.shape[0] is not X.shape[0]:
            raise ValueError("SampleWeight is of length %d instead of %d like X",
                              sampleWeight.shape[0], X.shape[0])
        elif np.any(sampleWeight < 0):
            raise ValueError("sampleWeight contains negative elements")
        else:
            w = np.array(sampleWeight)

        if len(np.unique(y)) != 2:
            raise ValueError(
                    """"%d label(s) detected, this implementation only 
                        handles binary classification 
                    """ % len(np.unique(y)))
        else:
        #The algorithm depends on labels being 0 and 1 so make a dummy list
        # with the right labels. 
            self.maxLabel = max(y)
            self.minLabel = min(y)
            Y = y
            Y[Y == self.maxLabel] = 1
            Y[Y == self.minLabel] = 0
        
        for t in range(self.T):
            p = w/sum(w)
            if not isDist(p):
            #Check if we still have a distribution.
            #At this point this should always be the case.
                print(np.any(p < 0))
                print(np.isclose(sum(p),1.0))
                raise ValueError("p is not a distribution")
                
            self.weakLearn[t].fit(X, Y, sample_weight=p)
            err = np.sum(p*abs(self.weakLearn[t].predict(X)-Y))
            self.beta[t] = err/(1-err)
            w = w*pow(self.beta[t], 1-abs(self.weakLearn[t].predict(X)-Y))
            

        self.thresh = 0.5*sum(np.log(1/self.beta))

    def predict(self,X):
        height = 0
        for t in range(self.T):
            height += np.log(1/self.beta[t])*self.weakLearn[t].predict(X)
        if height >= self.thresh:
            return self.maxLabel
        else:
            return self.minLabel


parser = argparse.ArgumentParser(
            description="""This program uses AdaBoost to (binary) classify the 
                        test data based upon the provided training data and 
                        reports the number of iterations and the error rate in
                        the following format:

                        Trails error 0.0 0.0.
                        
                        The two trailing 0's are added for consistency with 
                        the other two algorithms used when plotting.
                        Author: Daniel Vente <danvente@gmail.com> June 2016""",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("trails", type = int, 
                    help="Number of trails with which to train AdaBoost.")
parser.add_argument("--trainData", default = "../data/AdaTrain.dat", 
                    help="Location of the training data.")
parser.add_argument("--testData", default = "../data/AdaTest.dat", 
                    help = "Location of the test data.")
parser.add_argument("--log", help="Store answer to file instead of printing" )
parser.add_argument("-v", "--verbose", action="store_true", 
                    help="Increase verbosity to improve readability (for humans)" )
args = parser.parse_args()

a = AdaBoostClassifier(args.trails)

X, y = load_svmlight_file(args.trainData)
testX, testY = load_svmlight_file(args.testData)  

if X.shape[0] != y.shape[0]:
    raise ValueError("X has %d examples instead of %d like y", 
                        X.shape[0], y.shape[0])

if testX.shape[0] != testY.shape[0]:
    raise ValueError("testX has %d examples instead of %d like testY", 
                        testX.shape[0], testY.shape[0])

a.fit(X,y)
boostErr = sum([a.predict(testX[t])for t in range(testX.shape[0])]!=testY)

if args.verbose is False:
    output = "{0} {1} 0.0 0.0\n".format(a.T,boostErr/testX.shape[0])
else:
    output = "Trails: {0}\nError(%): {1}\nInconclusive(%): 0.0\nZeros(%): 0.0\n".format(
            a.T,
            boostErr/testX.shape[0])

if args.log is not None:
    with open(args.log,"a") as file:
        file.write(output)
else:
    print(output)

