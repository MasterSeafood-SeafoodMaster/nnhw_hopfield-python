import matplotlib.pyplot as plt
import numpy as np
import sys
import toolkit as tk
np.set_printoptions(threshold=sys.maxsize)

#train = tk.loadData('Bonus_Training.txt')
#test = tk.loadData('Bonus_Testing.txt')

train = tk.loadData('Bonus_Training.txt')
test = tk.loadData('Bonus_Testing.txt')

c, h, w = train.shape; n = h*w

hopfield = tk.HopfieldNetwork(n)
hopfield.train(train)

fig, ax = plt.subplots(2,len(train))
for i in range(len(test)):
	result = hopfield.recall(test[i].flatten())
	result = result.reshape(h, w)
	ax[0][0].set_ylabel('Train')
	ax[0][i].matshow(train[i])
	ax[1][0].set_ylabel('Test')
	ax[1][i].matshow(result)
	
plt.show()