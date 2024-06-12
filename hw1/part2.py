import numpy as np
from part1 import *

# Data from credit.txt
data = np.array([
    ['Tim', 'low', 'low', 'no', 'no', 'male', 'low'],
    ['Joe', 'high', 'high', 'yes', 'yes', 'male', 'low'],
    ['Sue', 'low', 'high', 'yes', 'no', 'female', 'low'],
    ['John', 'medium', 'low', 'no', 'no', 'male', 'high'],
    ['Mary', 'high', 'low', 'yes', 'no', 'female', 'high'],
    ['Fred', 'low', 'low', 'yes', 'no', 'male', 'high'],
    ['Pete', 'low', 'medium', 'no', 'yes', 'male', 'low'],
    ['Jacob', 'high', 'medium', 'yes', 'yes', 'male', 'low'],
    ['Sofia', 'medium', 'low', 'no', 'no', 'female', 'low']
])
headers = [ 'Name', 'Debt', 'Income', 'Married?', 'Owns_Property', 'Gender', 'Risk']

def print_tree(node, depth=0):
    '''
    Print out the decision tree.
    Input:
        node: the root of the tree or any subtree, an instance of the Node class.
        depth: the depth of the current node in the tree, an integer scalar.
    '''
    headers = [ 'Debt', 'Income', 'Married?', 'Owns_Property', 'Gender']
    if node is None:
        return
    
    prefix = "| " * depth

    if node.isleaf:
        print(prefix + "Leaf: Predicted Label =", node.p)
    else:
        print(prefix + '\033[94m' + "Attribute", headers[node.i] + '\033[0m')
        for value, child_node in node.C.items():
            print(prefix + "| " + '\033[92m' + "Value =", value + '\033[0m')
            print_tree(child_node, depth + 1)

test = np.array([
    ["low", "low", "no", "Yes", "Male"],
    ["low", "medium", "yes", "Yes", "Female"]
]).T

X = data[:, 1:-1].T
Y = data[:,-1]
t1 = Tree.train(X,Y)

Y[-1] = 'high'
t2 = Tree.train(X,Y)

print_tree(t1)
print(f"[Tom Risk, Ana Risk] = {Tree.predict(t1, test)}")

print('\n\n')
print("Task 2-2")

print_tree(t2)
print(f"[Tom Risk, Ana Risk] = {Tree.predict(t2, test)}")

print('''
By changing Sophias credit risk to ‘high', we swap the second node attribute from Married to Debt and vice versa for 
the third node. The original tree then needs the gender attribute to further split our training data, while the second tree gains no information by doing so.
In our original decision tree, the property attribute is not used.
In our second decision tree, the property and gender attributes are not used.
      ''')