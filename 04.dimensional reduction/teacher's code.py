#PCA
#IRIS ==> 4 Features ==> 3 Feature ===> 3D

import pandas as pd
import matplotlib.pyplot as plt

from sklearn import decomposition
from sklearn import preprocessing

iris = pd.read_csv('iris.csv')
le = preprocessing.LabelEncoder()
iris.variety = le.fit_transform(iris.variety) # mapping

X = iris.iloc[: , 0:4].values
y = iris.iloc[: , 4].values


pca = decomposition.PCA(n_components=3)
pca.fit(X)
X = pca.transform(X)

fig = plt.figure(1 , figsize= (8,8))
ax = fig.add_subplot( 111, projection='3d')

ax.scatter(X[y==0 , 0] , X[y==0 , 1] , X[y==0 , 2] , c='red')
ax.scatter(X[y==1 , 0] , X[y==0 , 1] , X[y==0 , 2] , c='green')
ax.scatter(X[y==2 , 0] , X[y==2 , 1] , X[y==2 , 2] , c='blue')

plt.show()

       



