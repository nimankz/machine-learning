import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

dataSet=pd.read_csv("iris.csv")
x = dataSet.iloc[:,0:4].values
y = dataSet.iloc[:,4].values
xTrain,xTest,yTrain,yTest=train_test_split(x,y, test_size= 0.2)#, random_state=10)
bestK=0
bestScore=0
for k in range(1,51,2):
    model=KNeighborsClassifier(n_neighbors=k)
    model.fit(xTrain,yTrain)

    pred=model.predict(xTest)
    accuracy= accuracy_score(pred,yTest)

    if accuracy>bestScore:
        bestK=k
        bestScore=accuracy

print(f"Best score:{bestScore}\nbest k: {bestK}  !")