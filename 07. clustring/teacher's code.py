##Clustring ====> KMean
#UnSupervised

from tkinter import CENTER
from pandas import DataFrame
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

Data = {
    'x': [67,25,34,22,27,33,33,31,22,35,34,54,57,43,50,57,59,52,65,47,49,48,35,33,44,45,38,43,51,46],
    'y': [51,79,51,53,78,59,74,73,57,69,75,32,40,47,53,36,35,58,59,50,25,20,14,12,20,5,29,27,8,7]
    }

df = DataFrame(Data , columns = ['x','y'])
plt.scatter(df['x'] , df['y'] , c='black', s = 50)
plt.show()

kmeans = KMeans(n_clusters= 3).fit(df)
centroids = kmeans.cluster_centers_
print(centroids)
plt.scatter(df['x'] , df['y'], c= kmeans.labels_.astype(float) ,s=50)
plt.scatter(centroids[:,0], centroids[:,1] , c='red', s=100)
plt.show()
###############################
# print(kmeans.labels_)

cluster_map = pd.DataFrame()
cluster_map['data_index'] = df.index.values
cluster_map["cluster"] = kmeans.labels_

indexes = cluster_map[cluster_map.cluster == 1]['data_index'].tolist()
print(indexes)

for x in indexes:
    print(df.iloc[ x:x+1, :])
    
##############################################

p = kmeans.predict([[30,30]])    
print(p)

################################
p = kmeans.predict( df.values )
from sklearn.metrics import accuracy_score
print('Accuracy is ', accuracy_score(p ,  kmeans.labels_))






