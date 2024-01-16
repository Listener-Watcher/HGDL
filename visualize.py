import matplotlib.pyplot as plt


import numpy as np



#fig,ax = plt.subplots()
kl = np.array([0.8241,0.8272,0.8588,0.8927,0.9612,1.1466,1.4362,1.8297,2.6431,4.0711])
kl2 = np.array([0.4321,0.381,0.6133,0.7007,0.8621,0.9637,1.5131,2.1378,4.1235,4.6785])
x = np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
plt.plot(x,kl,label="ACM")
plt.plot(x,kl2,label="DRUG")
plt.xlabel("edge drop rate",fontsize=12)
plt.ylabel("KL loss",fontsize=12)
plt.grid()
plt.legend(fontsize=10)
plt.show()
