#main file
import sys
import glob
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder # to change the cat and dog to ones and zeros
from sklearn.model_selection import train_test_split

sys.path.insert(0, 'C:/Users/Amit/Documents/PythonTutorial/CodeD') # to read inside your folder for import 

from ReadFiles import ReadFiles

# importing training data 

filePath="C:/Users/Amit/Documents/PythonTutorial/Kaggle_dogs_cats/train"

k=ReadFiles(filePath)
y=k.loadFile(filePath)
(data,labels)=y
le=LabelEncoder(); # you have to initiate this by own
labels=le.fit_transform(np.array(labels))
data=data.reshape((data.shape[0],32*32*3))
print("[yo!] feature matrix: {:.1f}MB".format(data.nbytes/(1024*1000.0)))

(dataTrain,dataTest,labelsTrain,labelsTest)=train_test_split(data,labels,test_size=0.25,random_state=42)



# importing test data

#filePath="C:/Users/Amit/Documents/PythonTutorial/Kaggle_dogs_cats/test_small"

#k=ReadFiles(filePath)
#y=k.loadFile(filePath)
#(dataTest,labelsTest)=y
#le=LabelEncoder(); # you have to initiate this by own
#labelsTest=le.fit_transform(np.array(labelsTest))
#dataTest=dataTest.reshape((dataTest.shape[0],32*32*3))
#print("[yo!] test matrix: {:.1f}MB".format(dataTest.nbytes/(1024*1000.0)))

#Filepath="C:/Users/Amit/Documents/PythonTutorial/Kaggle_dogs_cats/test_small"
#files=glob.glob(Filepath+"/*")
#i=0
#data=[]
#for file in files:
    #image1=cv2.imread(file)
    #image=cv2.resize(image1,(32,32),interpolation=cv2.INTER_AREA)
    #i+=1
    #print("reading test data {}/{} files".format(i,len(files)))
    #data.append(image)
#TestData=np.array(data)
##print(TestData.shape)
#TestData.reshape((TestData.shape[0],32*32*3))    
#print("[yo!] feature matrix: {:.1f}MB".format(TestData.nbytes/(1024*1000.0)))

# lets build up knn neighbor

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

model=KNeighborsClassifier(n_neighbors=100,n_jobs=-1)
model.fit(dataTrain,labelsTrain)

print(classification_report(labelsTest,model.predict(dataTest),target_names=le.classes_))
                        