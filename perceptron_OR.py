# check the perceptron models

from ML_tools import perceptron
import numpy as np

# construct OR dataset
#X=np.array([[0,0],[0,1],[1,0],[1,1]])
#y=np.array([[0],[1],[1],[1]])

# construct XOR
X=np.array([[0,0],[0,1],[1,0],[1,1]])
y=np.array([[0],[1],[1],[0]])
# initiate perceptron

p=perceptron.perceptron(X.shape[1],alpha=0.1)

p.fit(X,y,epochs=20)

print("test OR perciptron")

for (x,target) in zip(X,y):
    pred=p.predict(x)
    print("data={},ground truth={},pred={}".format(x,target[0],pred))
    #print(target)