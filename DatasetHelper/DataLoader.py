"""dataset loader where we will load images from a given directory""" 

import numpy as np
import cv2
import os
import glob

class DataLoader:
    def __init__(self,filepath,preprocessors=None):
        # store image preprocessors
        self.preprocessors=preprocessors
        self.filepath=filepath
        
        # check if there are more preprocessors if not initialize them as empty list
        if self.preprocessors is None:
            self.preprocessors=[]
    
    def load(self):
        files=glob.glob(self.filepath+"/*")
        i=0
        data=[]
        labels=[]
        for file in files:
            image=cv2.imread(file)
            #image=cv2.resize(image1,(32,32),interpolation=cv2.INTER_AREA)
            # check to see if preprocessors are there
            label1=os.path.splitext(os.path.basename(file))[-2]
            label=label1[0:3]
            i+=1            
            if self.preprocessors is not None: # basically i will initiate an preprocessing object and pass it here to resize the image sequentially
                for p in self.preprocessors:
                    image=p.PreProcessing(image)            
            data.append(image)
            labels.append(label)
        print("reading  {}/{} files".format(i,len(files)))
        return (np.array(data),np.array(labels))        