"""gradient descent """

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs     
import numpy as np
import argparse
import matplotlib.pyplot as plt

def sigmoid_activation(x):
    return 1.0/(1+np.exp(-x))

def predict(X,W):
    preds=sigmoid_activation(X.dot(W))
    # apply threshold 
    preds[preds<=0.5]=0
    preds[preds>0.5]=1
    return preds

def next_batch(X,y,batchsize):
    # loop over current dataset X in mini batches
    # and yield X and y in tuples
    for i in np.arange(0,X.shape[0],batchsize):
        yield (X[i:i+batchsize],y[i:i+batchsize])

# argument parser
ap=argparse.ArgumentParser()
ap.add_argument("-e","--epochs",type=float,default=100,help="# of epochs")
ap.add_argument("-a","--alpha",type=float,default=0.01,help=" learning rate")
ap.add_argument("-b","--batch size",type=int,default=32,help="batch size")
args=vars(ap.parse_args())

# generate a 2 class classification prob from blobs

(X,y)=make_blobs(n_samples=1000,n_features=2,centers=2,cluster_std=1.5,random_state=1)
print(y.shape,X.shape)
y=y.reshape((y.shape[0],1)) # need to do it with pythn else the dimensions are (1000,) not (1000,1)
print(y.shape)
print(X.shape)

# Insert a column of 1's as for bias term
X=np.c_[X,np.ones((X.shape[0]))]

print(X.shape)
#print(X[:,-1])

(trainX,testX,trainY,testY)=train_test_split(X,y,test_size=0.5,random_state=42)

# initialize weight matrix
W=np.random.randn(X.shape[1],1)
print("initial W",W)
print(W.shape)
print(W.T.shape)

# initialize loss
losses=[]

# loop over epochs

# case of normal gradient descent
# uncomment below
for e in np.arange(0,args["epochs"]):
    epochLoss=[]
    
    for (batchX,batchY) in next_batch(trainX,trainY,args["batch size"]):
        
        preds=sigmoid_activation(batchX.dot(W))
        error=preds-batchY
        loss=np.sum(error**2)
        epochLoss.append(loss)
    
        # gradient descent
        gradient=batchX.T.dot(error)
    
        # update the weighing matrix
        W += -args["alpha"]*gradient
    losses.append(np.average(epochLoss))    # If using Andrew NG guidebook we ought to multiply it with 0.5
    # display every 5th epoch
    if e==0 or (e+1)%5==0:
        print("Epoch= {},loss={:.7f}".format(int(e+1),loss))

#print("gradient is:",gradient.shape)
#print(gradient)

# check misclassifications
preds=predict(testX,W)
print(classification_report(testY,preds))
print("updated W:",W)
#print(np.sum(losses))

# stochastic gradient descent --batch programming

# create a next batch function


