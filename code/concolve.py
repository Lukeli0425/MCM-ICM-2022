import numpy as np

a = np.arange(0,20)
print(a)
n = 5
b = np.convolve(a,np.ones((3))/3,mode='same')
print(b)

a = np.array([[1,2],[3,4]])
print(a[0])

