# convolution from scratch
import numpy as np
import argparse
import cv2

# new installation

from skimage.exposure import rescale_intensity # to build our own convolution 

def convolve(image,K): # k is the kernel that moves on the image for convolution
    # grabing spatial dimensions
    (iH,iW)=image.shape[:2] # rows are height and columns are width
    (kH,kW)=K.shape[:2] # kernel height and width
    
    #use padding at the border so that spatial size are not reduced
    pad=(kW-1)//2 # remember we always use odd kernel sizes
    print(pad)
    # implement the padding around the image borders
    image=cv2.copyMakeBorder(image,pad,pad,pad,pad,cv2.BORDER_REPLICATE)
    print(image[:2])
    output=np.zeros((iH,iW),dtype="float")
    print(output[:])
    print(iH+pad)
    # applying convolution
    for y in np.arange(pad,iH+pad):
        for x in np.arange(pad,iW+pad):
            # extracting ROI  by extracting the center of the current
            #(x-y) cordinate
            roi=image[y-pad:y+pad+1,x-pad:x+pad+1]
            # perform convolution
            
            #k=(roi*K).sum()
            k=np.sum(roi*K)
            output[y-pad,x-pad]=k
            
    output=rescale_intensity(output,in_range=(0,255))
    output=(output*255).astype("uint8")
    return output


# argument parser
ap=argparse.ArgumentParser()
ap.add_argument("-i","--image",required=True,help="path to input image")
args=vars(ap.parse_args())

#constructing average blur filters

smallBlur=np.ones((7,7),dtype="float")*(1.0/(7*7))
k=smallBlur
# load image
image=cv2.imread(args["image"])
gray=cv2.cvtColor(image,cv2.COLOR_BGRA2GRAY)
print(image.shape[:])# access the tuple
print("Applying Kernel")

ConvolveOut=convolve(gray,k)
openCVout=cv2.filter2D(gray,-1,k)

cv2.imshow("original",gray)
cv2.imshow("convolve",ConvolveOut)
cv2.imshow("from library",openCVout)
cv2.waitKey(0)
cv2.destroyAllWindows()