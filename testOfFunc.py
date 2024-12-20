# import numpy as np
# from scipy import sparse
# import matplotlib
# matplotlib.use('TkAgg')  # Set an interactive backend
# import matplotlib.pyplot as plt
#
# x = np.array([[1,2,3],[4,5,6]])
# print("x:\n{}".format(x))
#
# eye = np.eye(4)
# print("\nNumPy array:\n", eye)
#
# sparse_matrix = sparse.csr_matrix(eye)
# print("\nSciPy sparse CSR matrix:\n", sparse_matrix)
#
# data = np.ones(4)
# row_indices = np.arange(4)
# col_indices = np.arange(4)
# eye_coo = sparse.coo_matrix((data,(row_indices, col_indices)))
# print("\nCOO representation:\n", eye_coo)
#
# xy = np.linspace(-10, 10, 100)
# y = np.sin(xy)
# plt.plot(xy,y,marker="x")
#
# plt.show()

# import pandas as pd
# from IPython.display import display, HTML, Image
#
# data = {'name': ["John", "Anna", "Peter", "Linda"], 'Location' : ["New York", "Paris", "Berlin", "London"], 'Age' : [24, 13, 53, 33]}
#
# data_pandas = pd.DataFrame(data)
#
# display(data_pandas)
#
# print("\nOnly records where age i > 30:\n")
#
# display(data_pandas[data_pandas.Age > 30])

import sys
print("Python version: ", sys.version)

import pandas as pd
print("pandas version: ", pd.__version__)

import matplotlib
print("matplotlib version: ", matplotlib.__version__)

import numpy as np
print("NumPy version: ", np.__version__)

import scipy as sp
print("SciPy version: ", sp.__version__)

import IPython
print("IPython version: ", IPython.__version__)

import sklearn
print("scikit-learn version: ", sklearn.__version__)