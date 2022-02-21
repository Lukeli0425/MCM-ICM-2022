from scipy import optimize
import numpy as np
from datetime import datetime,timedelta

c = -np.array([2,1])
A_ub =  np.array([[1,1]])
b_ub =  np.array([2])
x_b = (-1 ,None)
y_b = (-1 ,None)

res = optimize.linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds = (x_b,y_b))
print(res.x)

print(min(datetime.strptime('09-11-2016','%m-%d-%Y'),datetime.strptime('09-12-2016','%m-%d-%Y')))