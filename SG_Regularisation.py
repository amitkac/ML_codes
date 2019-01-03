from sklearn.linear_model import SGDClassifier  
#main file
import numpy as np
from sklearn.preprocessing import LabelEncoder # to change the cat and dog to ones and zeros
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from preprocessing import ImageProcessor
from DatasetHelper import DataLoader
import argparse
import sys

ap=argparse.ArgumentParser() # initilize the argument parser
ap.add_argument("-d","--dataset",required=True,help="path to dataset")
args=vars(ap.parse_args())

print("Loading images ....")
imagepath=args["dataset"]
sp=ImageProcessor.ImageProcessor(32,32)
sdl=DataLoader.DataLoader(imagepath,preprocessors=[sp])
(data,labels)=sdl.load()
data=data.reshape((data.shape[0],32*32*3))

print("loaded {:1f}MB".format(data.nbytes/(1024*1000.0)))

# create label encoder

le=LabelEncoder()
labels=le.fit_transform(labels)

(trainX,testX,trainY,testY)=train_test_split(data,labels,test_size=0.25,random_state=42)

# check different regularization results

for r in (None,'l1','l2'):
    # train the classifier using softmax loss function and specific regularization function
    # for 10 epochs
    print("Training model with '{}' penality".format(r))
    model=SGDClassifier(loss="log",penalty=r,max_iter=100,learning_rate="constant",eta0=0.01,random_state=42)
    model.fit(trainX,trainY)
    # evaluate classifier
    accuracy=model.score(testX,testY)
    print("with '{}' penalty, the accuracy is {:.2f}%".format(r,accuracy*100))
