import matplotlib.pyplot as plt
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)

f = open('Basic_Training.txt', 'r')
lines = f.readlines()

train=[]; strings=[]; nodes=[]
for line in lines:
	line = line.replace("\n", "")
	nodes=[]
	for i in range(len(line)):
		if line[i]==" ":
			nodes.append("-1")
		else:
			nodes.append("1")
	if len(nodes)==0:
		train.append(strings)
		strings=[]
	else:
		strings.append(nodes)
train.append(strings)
train = np.array(train, dtype=np.int64)

print(train)


