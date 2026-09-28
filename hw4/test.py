import numpy as np
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
x = np.arange(0, 11) # original sample points, [0, 1, 2, …, 9, 10]
y = np.sin(x)
t, c, k = interpolate.splrep(x, y, k=3)
print("t:", t)
print("c:", c)
print("k:", k)
x_new = np.arange(0, 10.3, 0.1)
# new sample points, [0, 0.1, 0.2, ….., 10, 10.1, 10.2]
f = interpolate.BSpline(t, c, k)
y_new = f(x_new)
plt.plot(x,y,'o',x_new, y_new)
plt.show()