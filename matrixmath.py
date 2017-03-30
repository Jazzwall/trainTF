#code for doing numpy and tf matrix multiplication

import numpy as np
import tensorflow as tf

# wx + b using numpy
w = np.array([[-.5, .2, .1],[.7,-.8,.2]])
x = np.array([[.2],[.5],[.6]])
b = np.array([[.1],[.2]])

wx = np.dot(w,x)
result = wx + b
print("RESULT---------")
print(result)


# xw + b using numpy (notice that the matrices are transposed)
# tensorflow uses this transposed form
wt = np.transpose(w)
print(wt)
xt = np.transpose(x)
print(xt)
bt = np.transpose(b)
print("b transpose is...")
print(bt)

xw = np.dot(xt, wt)
print("XW is...")
print(xw)
resulttranspose = xw + bt
print("Result transposed -------")
print(resulttranspose)