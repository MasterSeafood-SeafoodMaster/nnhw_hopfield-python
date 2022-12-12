import numpy as np

def loadData(path):
	f = open(path, 'r')
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

	return train

class HopfieldNetwork:
  def __init__(self, num_neurons):
    self.W = np.zeros((num_neurons, num_neurons))

  def train(self, input_patterns):
    for pattern in input_patterns:
      self.W += np.outer(pattern.flatten(), pattern.flatten())
    np.fill_diagonal(self.W, 0)

  def recall(self, initial_state, max_iter=10):
    state = initial_state
    for _ in range(max_iter):
      dot_product = np.dot(state, self.W)
      state = np.where(dot_product > 0, 1, -1)

    return state