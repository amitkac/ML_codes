#convolution network
# shallownet

from keras.models import Sequential   # feed forward
from keras.layers.convolutional import Conv2D # convolution layer
from keras.layers.core import Activation
from keras.layers.core import Flatten   #flattening to 1D array
from keras.layers.core import Dense     #connect the layers
from keras import backend as K  # always check the backend

class ShallowNet:
    
    @staticmethod
    def build(width,height,depth,classes):
        #initilize the model along with the input shape to 
        #channel last
        
        model=Sequential()
        inputShape=(height,width,depth)   # this is as per keras, remmeber in theano--the channels(depth comes first)
        
        #if we are using channel forst, update the input shape
        
        if K.image_data_format()=="channels_first":
            inputShape=(depth,height,width)    # this makes sure that whaetever backend we are using it is confirming to it
            
        #define the CONV=>RELU layer
        model.add(Conv2D(32,(5,5),padding="same",input_shape=inputShape))  # 32 filters each is of size 3x3, padding same means input image is same as output
        
        model.add(Activation("relu"))
        
        #softmax classifier
        model.add(Flatten()) # flatten the output to 1D after activation
        model.add(Dense(classes))  # connect the flatten to output with numbers equal to same classes
        model.add(Activation("softmax"))
        #print('inputShape')
        #return the model
        return model
    