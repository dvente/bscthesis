#!/usr/bin/python3
import numpy as np
import random as rand
import sys
import argparse

parser = argparse.ArgumentParser(description='generate 10-gaussian-data')
parser.add_argument('cases', metavar = 'N', type = int, help='Number of examples to generate')
parser.add_argument('file', default = "../generated/train.dat",help="location to store the generated data")

args = parser.parse_args()

data = []
for m in range(0,args.cases):
	vec = np.random.normal(0,1,10).tolist()
	if sum(map(lambda x: x**2, vec)) > 9.34:
		vec.append(1)
	else:
		vec.append(0)

	data.append(vec)

with open(args.file,'w') as file:
	file.write(str(sys.argv[1])+"\n")
	for i in data:
		for j in i:
			file.write(str(j)+" ")
		file.write("\n")
	