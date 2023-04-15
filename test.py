import numpy as np
x = np.random.rand(2, 4, 5, 6) # create a random array of shape (3, 4, 5, 6)

for i in range(x.shape[0]):
    x = x[i, ...]
    x = x.reshape(1, -1)
print(x)