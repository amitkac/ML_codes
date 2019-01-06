from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse


time=100 # to avoid conflict with epochs (keyword) we use time
# Constructing argument parser

ap=argparse.ArgumentParser()
ap.add_argument("-o","--output",required=True,help="path to save the plot")
args=vars(ap.parse_args())

print("loading CIFAR-10 dataset")
((trainX,trainY),(testX,testY))=cifar10.load_data()

trainX=trainX.astype("float")/255.0
testX=testX.astype("float")/255.0

# flattening out the data as 32*32*3
trainX=trainX.reshape((trainX.shape[0],32*32*3))
testX=testX.reshape((testX.shape[0],32*32*3))

# label hot encoding
lb=LabelBinarizer()
trainY=lb.fit_transform(trainY)
testY=lb.fit_transform(testY)

# Initialize label names
labelNames=["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

# defining network architecture--3072-1024-512-10

model=Sequential()
model.add(Dense(1024,input_shape=(3072,),activation="relu"))
model.add(Dense(512,activation="relu"))
model.add(Dense(10,activation="softmax"))

print("training network")
sgd=SGD(0.01)
model.compile(loss="categorical_crossentropy",optimizer=sgd,metrics=["accuracy"]) # use cross entropy as loss metric and the reason to change the int to vector
H=model.fit(trainX,trainY,validation_data=(testX,testY),epochs=time,batch_size=32) # remember batch size makes it the back propagation 

# network evaluation
print("Evaluating network")
predictions=model.predict(testX,batch_size=32)
# each datapoint will have 10 probabilities associated with it 
#In MNIST we have 17,500 data so with eah having 10 probabilities it is (17,500,10)size
# what we need is to have label with maximum probability along columns
#that's the reason we take axis=1 and argmax to find the maximum probabilityand then we fetch the index
# which is finally our class label==> .argmax(axis=1) gives the index
# also testY is hot encoded with vectors of size 10
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1),target_names=labelNames))

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
                      