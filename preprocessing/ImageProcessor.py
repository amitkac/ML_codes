""" creating the file processing submodule to track the code"""

import cv2

class ImageProcessor:
    def __init__(self,width,height,inter=cv2.INTER_AREA): # we will keep it generic in case someone has to use a different interpolation method
        #fetch the dimensions of the image we want and resize them so that our method works uniformaly
        self.width=width
        self.height=height
        self.inter=inter
        # we initiated all the variables here
    
    def PreProcessing(self,image): # here we need the image and the dimensions to which we need to resize the image
        return cv2.resize(image,(self.width,self.height),interpolation=self.inter)
        # we will fetch the image and resize it back, the image we get is will be in numpy format 
        