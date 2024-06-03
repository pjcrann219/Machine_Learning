import math
import numpy as np
from collections import Counter
from part1 import *

t = Node(None,None) 
t.isleaf = False 
t.i = 1
c1 = Node(None,None)
c2 = Node(None,None)
c1.isleaf= True
c2.isleaf= True

c1.p = 'c1' 
c2.p = 'c2' 
t.C = {'high':c1, 'low':c2}

X = np.array([['apple','apple','apple','banana'],
                ['high','low','low','high'],
                ['a','b','c','a']])
Y = Tree.predict(t,X)

assert type(Y) == np.ndarray
assert Y.shape == (4,) 
assert Y[0] == 'c1'
assert Y[1] == 'c2'
assert Y[2] == 'c2'
assert Y[3] == 'c1'
