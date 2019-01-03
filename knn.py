# create a knn neighbour file"

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from preprocessing import ImageProcessor
from DatasetHelper import DataLoader
import argparse
import sys

#sys.path.insert(0, 'C:/Users/Amit/Documents/PythonTutorial/CodeD/preprocessing')
#sys.path.insert(0, 'C:/Users/Amit/Documents/PythonTutorial/CodeD/DatasetHelper')

# create argument parser wher we will parse dataset to the code and 
# use the classification method to work out the result 

ap=argparse.ArgumentParser() # initilize the argument parser
ap.add_argument("-d","--dataset",required=True,help="path to dataset")
ap.add_argument("-k","--neighbors",type=int,default=1,help="enter number of neighbors wanted")
ap.add_argument("-j","--jobs",type=int,default=-1,help="number of jobs for k-nn algol(-1 is use all cores)")

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

print("begin k-nn classifier")

model=KNeighborsClassifier(n_neighbors=args["neighbors"],n_jobs=args["jobs"])
model.fit(trainX,trainY)
print(classification_report(testY,model.predict(testX),target_names=le.classes_))
          