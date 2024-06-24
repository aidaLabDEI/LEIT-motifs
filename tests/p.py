import numpy as np

a = np.array([[7, 9, 1], 
             [5, 8, 7], 
             [7, 8, 0],
             [1, 4, 5]])
a = np.sort(a, axis=0)

b = np.array([1,2,3,4,5])
print(b[:3])