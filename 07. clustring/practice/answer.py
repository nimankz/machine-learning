# dataset iris  bedone estefade az setone variety aval clustring ra be 3 goroh
# bad be 4 goroh va deghat harkodam ra andaze begirim va bebinim kodam dadghgightar ast 

##Clustring ====> KMean
#UnSupervised

from tkinter import CENTER
from pandas import DataFrame
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Data = {
#     'x': [67,25,34,22,27,33,33,31,22,35,34,54,57,43,50,57,59,52,65,47,49,48,35,33,44,45,38,43,51,46],
#     'y': [51,79,51,53,78,59,74,73,57,69,75,32,40,47,53,36,35,58,59,50,25,20,14,12,20,5,29,27,8,7]
#     }

# df = DataFrame(Data , columns = ['x','y'])



###################### PCA

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn import preprocessing

iris = pd.read_csv('iris.csv')
le = preprocessing.LabelEncoder()
iris.variety = le.fit_transform(iris.variety) # mapping

X = iris.iloc[: , 0:4].values
y = iris.iloc[: , 4].values

pca = decomposition.PCA(n_components=2)
pca.fit(X)
X = pca.transform(X)

plt.scatter(X[y==0 , 0] , X[y==0 , 1] , c='red')
plt.scatter(X[y==1 , 0] , X[y==0 , 1] , c='green')
plt.scatter(X[y==2 , 0] , X[y==2 , 1] , c='blue')

plt.show()


###########################

df = pd.read_csv('iris.csv')

plt.scatter(X[: , 0] , X[: , 1] , c='black', s = 50)
plt.show()

for groupnumber in range(3,4):
    kmeans = KMeans(n_clusters= groupnumber ).fit(X)
    centroids = kmeans.cluster_centers_
    print(kmeans.labels_.astype(float))
    print(centroids)
    plt.scatter(X[: , 0], X[: , 1] , c= kmeans.labels_.astype(float) ,s=50)
    plt.scatter(centroids[:,0], centroids[:,1] , c='red', s=100)
    plt.show()
    p = kmeans.predict( X )
    from sklearn.metrics import accuracy_score
    print(y)
    print(p)
    print('Accuracy is ', accuracy_score(p ,  y )) #kmeans.labels_))
    input('????')




