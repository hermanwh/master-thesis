import numpy as np
import matplotlib.pyplot as plt

nrOfPoints = 8000
periods = 8
x = np.linspace(0, periods*np.pi, nrOfPoints)

f = 0.1*np.sin(10*x)
g = 0.01*np.sin(50*x)
h = np.sin(x)
i = 5*np.sin(x)

j = f + g + h
k = f + g + i

print(int(nrOfPoints/periods))
y = np.concatenate([j[0:int(6*nrOfPoints/periods)], k[int(6*nrOfPoints/periods):int(7*nrOfPoints/periods)], j[int(7*nrOfPoints/periods):]])

plt.plot(x, y)
plt.show()

