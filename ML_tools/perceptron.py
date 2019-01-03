# perceptron
import numpy as np

class perceptron:
    def __init__(self,N,alpha=0.01):
    # initialize the weight matrix with random values
        self.W=np.random.randn(N+1)/np.sqrt(N) # one entry is for bias, the division is to scale the weight matrix for faster convergence
        self.alpha=alpha
    
    def step(self,x):
    # apply step logic
        return 1 if x>0 else 0

    def fit(self,X,y,epochs=10):
    # add a column of 1's which represent the bias
        X=np.c_[X,np.ones((X.shape[0]))]
        # loop over the epochs
        for epoch in np.arange(0,epochs):
            # find W.X and pass it to step
            for (x,target) in zip(X,y):
                # take the dot product bw imput and weight
                # pass it to step for predictions
                p=self.step(np.dot(x,self.W))
                # perform weight update if predictions doesn't meet target
                if p != target:
                    error=p-target
                    
                    #update weight
                    self.W+=-self.alpha*error*x
            
    def predict(self,X,addBias=True):
        X=np.atleast_2d(X)
        X=np.c_[X,np.ones((X.shape[0]))]
       # find dot product and send it thru step function
        return self.step(np.dot(X,self.W))       
       
    # ensure our input is a matrix
       
    # check if bias is added, if not add it
       


    