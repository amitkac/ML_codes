#image to array processor using keras

from keras.preprocessing.image import img_to_array


class ImageToArrayPreprocessor:
    
    def __init__(self,dataformat=None):
        # store the image dataformat
        self.dataformat=dataformat
    
    def PreProcessing(self,image):
        # apply keras utility that arranges the dimensions
        # properly
        return img_to_array(image,data_format=self.dataformat)
    