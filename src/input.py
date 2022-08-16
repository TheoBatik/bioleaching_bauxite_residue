# Purpose of script

import os
import numpy as np

# Move to data folder
os.chdir('../data')

# Input the stoichiometric matrices, A and B.

A = []
B = []

with open('A.csv', 'r') as file:
    for line in file: 
        line = line.strip()
        line = line.split(',')
        A.append([int(item) for item in line])

with open('B.csv', 'r') as file:
    for line in file: 
        line = line.strip()
        line = line.split(',')
        B.append([int(item) for item in line])

A = np.asarray(A)
B = np.asarray(B)

# Define the stoichiometric matrix, X.

X = B - A
# X = np.vstack((X, -X)) # stacks the reverse reactions (-X) onto the forward reaction (X)
print(X)
