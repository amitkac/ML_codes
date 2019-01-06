#main neuralneteork

import numpy as np
from ML_tools.NeuralNw import NeuralNetwork


#contruct XOR dataset
X=np.array([[0,0],[0,1],[1,0],[1,1]])
y=np.array([[0],[1],[1],[0]])
nn=NeuralNetwork([2,2,1],alpha=0.01)
nn.fit(X,y,epochs=200000)

#predict

for (x,target) in zip(X,y):
    pred=nn.predict(x)[0][0]
    step=1 if pred>0.5 else 0
    print("data={},ground_truth={},pred={:.4f},step={}".format(x,target[0],pred,step))
    