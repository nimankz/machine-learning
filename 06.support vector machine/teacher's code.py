

import pandas as pd
import matplotlib.pyplot as plt

from sklearn import svm

x1 = [1, 8,1.5 ,8.7,1  ,9 ,2,10  ,2.3,9]
y1 = [2, 8,1.8 ,8.3,0.6,11,2,8.8 ,3.1,7.8]

plt.scatter(x1, y1, c= 'red')
plt.show()

y = ['A','B','A','B','A','B','A','B','A','B']

X = list(zip(x1,y1))
print(X)
print(y)

model = svm.SVC(kernel= 'linear')
model.fit(X, y)

p = model.predict([[1,2]])
print('prediction of [1,2] is ' ,p)
p = model.predict([[11,12]])
print('prediction of [11,12] is ' ,p)

import numpy as np
xx = np.linspace(0,11 ,40)
print(xx)

w =model.coef_[0]
print(w[0])
print(w[1])
print(model.intercept_[0])

yy = (-w[0]/w[1]) * xx  - model.intercept_[0] / w[1]
print('_' * 50)
print(xx)
print(yy)

plt.plot(xx ,yy ,c='b')
plt.scatter(x1 ,y1 ,c='g')
plt.show()
















