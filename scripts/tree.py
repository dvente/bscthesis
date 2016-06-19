import numpy as np
import argparse

from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_svmlight_file


parser = argparse.ArgumentParser(
        description=""" This program fits and tests a simple decision tree on 
                        the provided data and reports its error on the test 
                        data, used as reference with the other algorithms.
                        Author: Daniel Vente <danvente@gmail.com> June 2016
                        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
        "-n", "--nodes", type = int, default = 0, 
        help="Maximum number of nodes of the tree, use 0 for unlimited nodes.")
parser.add_argument(
        "--trainData",  default = "../data/AdaTrain.dat",
        help="Location of the training data.")
parser.add_argument(
        "--testData", default = "../data/AdaTest.dat", 
        help = "Location of the test data." )

args = parser.parse_args()

#boostErr = 0
X, y = load_svmlight_file(args.trainData)

if(args.nodes == 0):
    tree = DecisionTreeClassifier()
else:
    tree = DecisionTreeClassifier(max_depth = max(np.log2(args.nodes),1))

tree.fit(X,y)

testX, testY = load_svmlight_file(args.testData)
boostErr = np.sum(np.equal(
        np.array([tree.predict(testX[t]) for t in range(testX.shape[0])]).T,
        testY))

print(tree.tree_.node_count, boostErr/testX.shape[0])   


