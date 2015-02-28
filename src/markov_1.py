import numpy as np
from scipy.linalg import toeplitz
from scipy import linalg as LA
from numpy import *


#roe = 0.9543
#roe = 0.9543
roe=0.9999
roe_list = []
#data_dimension = 49 
data_dimension = 64

for i in range (data_dimension):
    roe_list.append(roe**i)

roe_list = np.array(roe_list)

covariance_matrix = toeplitz(roe_list, roe_list)

e_vals, e_vecs = LA.eig(covariance_matrix)

#DataOut = e_vecs
#savetxt('markov_1.dat', DataOut)

DataOut = e_vecs
savetxt('t.txt', DataOut)

savetxt('cov.txt',covariance_matrix)
