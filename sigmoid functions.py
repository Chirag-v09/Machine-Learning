import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

x = np.arange(-10,10,0.01)

sig = 1/(1+np.power(np.e,-x))
sig_1 = np.power(np.e,-x)/(1+np.power(np.e,-x))

plt.plot(x,sig)
plt.plot(x,sig_1)
plt.show()