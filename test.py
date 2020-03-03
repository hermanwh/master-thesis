


"""

import requests

URL = "http://factpages.npd.no/Default.aspx?culture=nb-no&nav1=field&nav2=TableView%7cProduction%7cSaleable%7cMonthly"
URL = "https://www.ntnu.no/studier/emner/TDT4100"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:66.0) Gecko/20100101 Firefox/66.0",
    "Accept-Encoding": "*",
    "Connection": "keep-alive"
}


print(URL)

page = requests.get(URL, headers={'Accept-Encoding': None})

pprint(asd)

"""


"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

Y = np.array([0, 2, 1, 3, 5, 4, 7, 8, 6])
X = list(range(Y.shape[0]))

print(Y)
print(X)

print(len(Y))
print(len(X))

polynomialCoefficients = np.polyfit(X, Y, 3)
print(polynomialCoefficients)

polynomal = np.poly1d(polynomialCoefficients)
print(polynomal)

X_fine = np.linspace(X[0], X[-1], 1000)
print(X_fine)

func_vals = polynomal(X_fine)
print(func_vals)

fig,ax = plt.subplots()
ax.set_xlabel('X-verdi')
ax.set_ylabel('Y-verdi')
ax.set_title('Et plott')

ax.plot(X, Y, label='Plot of actual points', color="black")
ax.plot(X_fine, func_vals, label='Interpolated values', color="darkgreen")

func_vals = polynomal(X)
sinefunc = np.sin(X)
print(sinefunc)
print(func_vals)
joined = np.concatenate((sinefunc.reshape(-1, 1), func_vals.reshape(-1, 1)), axis=1)

model = LinearRegression().fit(joined, Y)


pred = model.predict(np.concatenate((np.sin(X_fine).reshape(-1, 1), polynomal(X_fine).reshape(-1, 1)), axis=1))

ax.plot(X_fine, pred, label='Prediction', color="red")

plt.show()

"""