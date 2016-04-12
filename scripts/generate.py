#!/usr/bin/python3.4
import numpy as np
import random as rand
import sys
import argparse

parser = argparse.ArgumentParser(description='generate 10-gaussian-data')
parser.add_argument('cases', metavar = 'N', type = int, help='Number of examples to generate')
parser.add_argument('file', default = "../generated/train.dat",help="location to store the generated data")
parser.add_argument('--label', '-l', choices=['0','1'], default = 0, help="give output in {0,1} or {-1,+1}  resp ")

args = parser.parse_args()

data = []
for m in range(0,args.cases):
	vec = np.random.normal(0,1,10).tolist()
	if sum(map(lambda x: x**2, vec)) > 9.34:
		vec.append(1)
	else:
		if(args.label == '0'):
			vec.append(0)
		elif(args.label == '1'):
			vec.append(-1)

	data.append(vec)

with open(args.file,'w') as file:
	file.write(str(args.cases)+"\n")
	for i in data:
		for j in i:
			file.write(str(j)+" ")
		file.write("\n")
	