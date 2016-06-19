import numpy as np
import argparse

from sklearn.datasets import dump_svmlight_file

parser = argparse.ArgumentParser(
        description="""Generates gaussian data like described in the thesis, 
        and stores it in the provided location. The data consists of 10 
        independant Gaussians and their label is 1 if their sum is more 
        than 9.34.
        Author: Daniel Vente <danvente@gmail.com> June 2016
        """)
parser.add_argument("cases", type = int, help="Number of examples to generate")
parser.add_argument("file", help="Location to store the generated data.")
args = parser.parse_args()

X = []
y = []
for m in range(args.cases):
    vec = np.random.normal(0,1,10).tolist()
    if sum(map(lambda x: x**2, vec)) > 9.34:
        y.append(1)
    else:
        y.append(-1)

    X.append(vec)

dump_svmlight_file(X,y,args.file)
