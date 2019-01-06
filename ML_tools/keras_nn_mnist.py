# importing packages

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import argparse 

# new ones we have not seen yet
from sklearn.preprocessing import LabelBinarizer # use to hot encode labels as vectors
from sklearn import datasets
from keras.models import Sequential # means the network will be feedforward
from keras.layers.core import Dense  # implementation of fully connected layer

from keras.optimizers import SGD    # use SGD to optimize the network

time=50 #avoiding conflict with epochs
# constructing argument parser

ap=argparse.ArgumentParser()
ap.add_argument("-o","--output",required=True,help="path to the output loss/accuracy plot")
args= vars(ap.parse_args())

# loading MNIST dataset
print("loading dataset...")
#dataset=datasets.fetch_mldata("MNIST Original")
dataset = datasets.fetch_mldata('MNIST original', transpose_data=True, data_home='files')
# normalizing  raw pixels intensities to the range [0.0,1.0]
data=dataset.data.astype("float")/255.0

# splitting data
(trainX,testX,trainY,testY)=train_test_split(data,dataset.target,test_size=0.25)

# convert the labels from integers to vectors (hotspot encoding)
lb=LabelBinarizer()
trainY=lb.fit_transform(trainY)
testY=lb.fit_transform(testY)

# defining the neural network layers from input to output
# the model is 784-256-128-10
model=Sequential()
model.add(Dense(256,input_shape=(784,),activation="sigmoid"))
model.add(Dense(128,activation="sigmoid")) # remember in backpropagation, it has to be easily differentiable--that;s why
model.add(Dense(10,activation="softmax")) # to get the result as a normalized probability distribution

# using the trainign method= SGD
sgd=SGD(0.01)
model.compile(loss="categorical_crossentropy",optimizer=sgd,metrics=["accuracy"]) # use cross entropy as loss metric and the reason to change the int to vector
H=model.fit(trainX,trainY,validation_data=(testX,testY),epochs=time,batch_size=128) # remember batch size makes it the back propagation 
# In reality it is a wrong practice to use test data as validation data---it has to be different than test and training which the model has not seen before.

# network evaluation
print("Evaluating network")
predictions=model.predict(testX,batch_size=128)
# each datapoint will have 10 probabilities associated with it 
#In MNIST we have 17,500 data so with eah having 10 probabilities it is (17,500,10)size
# what we need is to have label with maximum probability along columns
#that's the reason we take axis=1 and argmax to find the maximum probabilityand then we fetch the index
# which is finally our class label==> .argmax(axis=1) gives the index
# also testY is hot encoded with vectors of size 10
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1),target_names=[str(x) for x in lb.classes_]))

# plot the training loss and accuracy 
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,time), H.history["loss"], label="train_loss")
plt.plot(np.arange(0,time), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0,time), H.history["acc"], label="train_acc")
plt.plot(np.arange(0,time), H.history["val_acc"], label="val_acc")
plt.title("Training loss and accuracy")
plt.xlabel("Epoch");
plt.ylabel("loss/accuracy")
plt.legend()
plt.savefig(args["output"])

