from time import time
import numpy as np
from copy import deepcopy

x = [-0.123, 0.12, 0.12]
print(np.linalg.norm(x))
n = 1
t1 = time()
for i in range(n):
    norm1 = x / np.linalg.norm(x)
    print(norm1)
t2 = time()
v = x
for i in range(n):
    m = abs(max(x))
    if m <= 1:
        break
    v = deepcopy(x)
    for i in range(len(v)):
        v[i] /= m
print(v)
t3 = time()
print(t2 - t1)
print(t3 - t2)
#print(t4 - t3)
