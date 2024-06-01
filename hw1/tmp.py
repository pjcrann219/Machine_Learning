import math
import numpy as np
from collections import Counter
from part1 import *

X = np.array([['apple','orange','pineapple','banana'],
                ['high','high','low','low'],
                ['a','b','c','d']])
Y = np.array(['good','bad','okay','perfect'])
C = Tree.split(X,Y,1)

assert type(C) == dict
assert len(C) == 2 

assert isinstance(C['high'], Node)
assert isinstance(C['low'], Node)

assert C['high'].X.shape == (3,2)
assert C['high'].Y.shape == (2,)
assert C['high'].i == None 
assert C['high'].C == None 
assert C['high'].isleaf == False 
assert C['high'].p == None 

assert C['high'].X[0,0] == 'apple'
assert C['high'].X[0,1] == 'orange'
assert C['high'].X[1,0] == 'high'
assert C['high'].X[1,1] == 'high'
assert C['high'].X[2,0] == 'a'
assert C['high'].X[2,1] == 'b'

assert C['low'].X.shape == (3,2)
assert C['low'].Y.shape == (2,)
assert C['low'].i == None 
assert C['low'].C == None 
assert C['low'].isleaf == False 
assert C['low'].p == None 

assert C['low'].X[0,0] == 'pineapple'
assert C['low'].X[0,1] == 'banana'
assert C['low'].X[2,0] == 'c'
assert C['low'].X[2,1] == 'd'