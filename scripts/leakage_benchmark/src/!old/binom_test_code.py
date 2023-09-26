import numpy as np
from scipy.stats import binom

n = 20
p = 0.5

# print(n*p)
# x = [binom.pmf(i, n, p) for i in range(1, n+1)]
# print(x)
# print(np.mean(x))
print(binom.pmf(0, 8, 0.5))
