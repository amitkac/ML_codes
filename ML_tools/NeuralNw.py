# creating a neural networ
import numpy as np

class NeuralNetwork:
    def __init__(self,layers,alpha=0.01):
        #initialixe weight matrix and then store the network 
        #architecture and learning rate accordingly
        self.W=[]
        self.alpha=alpha # learning rate in neural ntwork to update the weights 
        self.layers=layers
        #intialize the wiight matrix-- it is layer[i]+1 (bias) to next layer-layer[i+1]
        
        for i in np.arange(0,len(layers)-2): # bcoz last layer is output and the previous one before it is connecting all to it
            w=np.random.randn(layers[i]+1,layers[i+1]+1)
            self.W.append(w/np.sqrt(layers[-2])) # normalizing with the elements in the final layer 
            
    def __repr__(self):
        # pythons debugging function
        return "neural nw:{}".format("-".join(str(l) for l in self.layers))
    
    def sigmoid(self,x): # sigmoid activation function
        return 1.0/(1+np.exp(-x))
    
    def sigmoid_deriv(self,x):
        #return self.sigmoid(x)*self.sigmoid(1-x)
        return x*(1-x)
        
    # training phase
    def fit(self,X,y,epochs=1000,displayUpdate=10000):
        #insert columns of 1 in the input data as a bias
        X=np.c_[X,np.ones((X.shape[0]))]
        
        # loop over the epochs
        for epoch in np.arange(0,epochs):
            #loop pver individual data point and train network
            for (x,target) in zip(X,y):
                self.fit_partial(x,target)
                
                #check to see if we should display the training update
                if epoch==0 or (epoch+1)%displayUpdate==0:
                    loss=self.calculate_loss(X,y)
                    print("Epoch={}, loss={:.7f}".format(epoch+1,loss))
    
    def fit_partial(self,x,y):
        #construct set of output activations for each layer
        # as our data point flows thru the network
        A=[np.atleast_2d(x)] # initialization of 2d array 
        
        #FEEDFORWARD method
        #loop over layers in the network
        for layer in np.arange(0,len(self.W)):
            #feedforward the activation of the current layer by
            #taking dot product of weights and activation to make it input
            #to next layer
            net=A[layer].dot(self.W[layer])
            
            #compute the net output by sigmoid function
            out=self.sigmoid(net)
            
            #once we have the outputs, add it to list of 
            #activations
            A.append(out)
            # so the last entry in A is the output or prediction 
            
        #BACKPROPAGATION
        #it starts with the last layer or the final output
        # and then goes back to the first layer
            error=A[-1]-y
        
        # Now from now onwards , we apply the chain rule to find
        #deltas----chain rule----
        
            D=[error*self.sigmoid_deriv(A[-1])]
            
        for layer in np.arange(len(A)-2,0,-1):
            delta=D[-1].dot(self.W[layer].T)
            delta=delta*self.sigmoid_deriv(A[layer])
            D.append(delta)            
            
                
                
        
        # deltas in every layer
        #the last 
        
        #since we looped backwards, we need to reverse the deltas
        D=D[::-1] # reverse a list
        
        # WEIGHT UPDATE
        #loop over laters
        for layer in np.arange(0,len(self.W)):
            self.W[layer]-=self.alpha*A[layer].T.dot(D[layer])
                        
            
                
    def predict(self,X,addbias=True):
        #initialize the output prediction as the input features
        p=np.atleast_2d(X)
        
        if addbias:
            p=np.c_[p,np.ones((p.shape[0]))]
        
        #loop over layers in the network
        for layer in np.arange(0,len(self.W)):
            p=self.sigmoid(np.dot(p,self.W[layer]))
            
        #return the value
        return p
    
    def calculate_loss(self,X,targets):
        targets=np.atleast_2d(targets)
        predictions=self.predict(X,addbias=False)
        loss=0.5*np.sum((predictions-targets)**2)
        
        return loss