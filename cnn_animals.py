# shallownet on animals

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from preprocessing import ImageProcessor
from preprocessing import ImageToArrayPreprocessor
from DatasetHelper import DataLoader
from ML_tools import ShallowNet

from keras.utils.np_utils import to_categorical

from keras.optimizers import SGD
#from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse

time=50

ap=argparse.ArgumentParser() # initilize the argument parser
ap.add_argument("-d","--dataset",required=True,help="path to dataset")
args=vars(ap.parse_args())

print("Loading images ....")
imagepath=args["dataset"]

sp=ImageProcessor.ImageProcessor(32,32)
iap=ImageToArrayPreprocessor.ImageToArrayPreprocessor()

sdl=DataLoader.DataLoader(imagepath,preprocessors=[sp, iap])
(data,labels)=sdl.load()
# data=data.reshape((data.shape[0],32*32*3))

le=LabelEncoder()
labels=le.fit_transform(labels)
#data=data.reshape((data.shape[0],32*32*3))
data=data.astype("float")/255.0
print(data.shape[:])


(trainX,testX,trainY,testY)=train_test_split(data,labels,test_size=0.1,random_state=42)
print(trainY.shape[:])
nb_classes = np.max(trainY) + 1
#print(data.shape[:])
#vectorize the labels 

trainY = to_categorical(trainY, num_classes = nb_classes)
testY = to_categorical(testY, num_classes = nb_classes)

print(nb_classes)
#trainY=LabelBinarizer().fit_transform(trainY)
#testY=LabelBinarizer().fit_transform(testY)
print(trainY.shape[:])
print("compiling model")
opt=SGD(lr=0.1)
model=ShallowNet.ShallowNet.build(width=32,height=32,depth=3,classes=nb_classes)
#print(model)

model.compile(loss="binary_crossentropy",optimizer=opt,metrics=["accuracy"])

print("training model")

H=model.fit(trainX,trainY,validation_data=(testX,testY),epochs=time,batch_size=10,verbose=1)

print('inputShape')
print("Evaluating network")
predictions=model.predict(testX,batch_size=10)

print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1),target_names=["cat","dog"]))

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