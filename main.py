import numpy as np
import matplotlib.pylab as plb
import matplotlib.pyplot as plt
import sys
sys.path.append('./ActivationFunction')
import sigmoid as sig

x1 = np.arange(0, 6, 0.1)
sin = np.sin(x1)
plt.plot(x1, sin)
plt.show()

x2 = np.arange(-5.0, 5.0, 0.1)
sigmoid = sig.sigmoid(x2)
plb.plot(x2,sigmoid)
plb.ylim(-0.1, 1.1)
plb.show()