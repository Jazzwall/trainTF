#code for doing numpy and tf matrix multiplication

import numpy as np
import tensorflow as tf

# wx + b using numpy
w = np.array([[-.5, .2, .1],[.7,-.8,.2]])
print(w)

x = np.array([[.2],[.5],[.6]])
print(x)

b = np.array([[.1],[.2]])
print(b)

wx = np.dot(w,x)
result = wx + b
print("RESULT---------")
print(result)


# xw + b using numpy (notice that the matrices are transposed)
# tensorflow uses this transposed form