import math
import numpy as np
from collections import Counter
from part1 import *

y = np.array([0.,1., 0., 1.])
e = Tree.entropy(y)
print(e)

y = np.array(['orange','apple','banana','pineapple'])
e = Tree.entropy(y)
print(e)
assert np.allclose(e, 2., atol = 1e-3) 