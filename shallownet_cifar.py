# shallownet on cifar 10

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.datasets import cifar10

from preprocessing import ImageProcessor
from preprocessing import ImageToArrayPreprocessor
from DatasetHelper import DataLoader
from ML_tools import ShallowNet
from keras.optimizers import SGD

import matplotlib.pyplot as plt
import numpy as np
import argparse 


time=20


ap=argparse.ArgumentParser()
ap.add_argument("-m","--model",required=True,help="path to save the model")

print("Loading cifar10 ....")
((trainX,trainY),(testX,testY))=cifar10.load_data()

# normalize test data
trainX=trainX.astype("float")/255.0
testX=testX.astype("float")/255.0

# convert label binalizer
lb=LabelBinarizer()
trainY=lb.fit_transform(trainY)
testY=lb.fit_transform(testY)

labelName=["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

print("compiling model")
opt=SGD(lr=0.01)
model=ShallowNet.ShallowNet.build(width=32,height=32,depth=3,classes=10)
#print(model)

model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["accuracy"])

print("training model")

H=model.fit(trainX,trainY,validation_data=(testX,testY),epochs=time,batch_size=32,verbose=1)


# saving the model
print("serializing model")
model.save(args["model"])

print("Evaluating network")
predictions=model.predict(testX,batch_size=32)

print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1),target_names=labelName))

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
plt.show()