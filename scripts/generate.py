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
