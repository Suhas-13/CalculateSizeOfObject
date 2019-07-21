from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils
from keras.models import *
from keras.layers import *
import numpy as np
import cv2
from types import MethodType
from PIL import Image
import random
from keras import layers
import keras.backend as K
import keras
random.seed(0)
global USERNAME
from keras.models import load_model
#Prediction code rewritten in only keras, no modules, except Keras, TF, CV2, PIL, Random, Types

USERNAME="suhas"
colors = [  ( random.randint(0,255),random.randint(0,255),random.randint(0,255)   ) for _ in range(5000)  ]

#latest_weights="C:/users/"+USERNAME+"/documents/flask/model.108"


#model=load_model("/users"+"/"+USERNAME+"/documents/CardModel.h5")


model2=load_model("/users"+"/"+USERNAME+"/documents/CardModel.h5")#model=resnet50_unet(256,256,256)
#model.load_weights(latest_weights)
#Loads model and weights based on earlier defined layers
def make_prediction(image,model):
    #Resizes input image to 256,256
    img = cv2.resize(image, ( 256,256))
    #Converts image to np float32 format
    img = img.astype(np.float32)
    img[:,:,0] -= 103.939
    img[:,:,1] -= 116.779
    img[:,:,2] -= 123.68
    img = img[ : , : , ::-1 ]

    
    #Generates prediction using loaded weights and loaded model
    pr = model.predict( np.array([img]) )[0]
    #Resapes output to 128,128
    pr = pr.reshape(( 128,128,256 ) ).argmax( axis=2 )
    #Creates seg_img numpy array
    seg_img = np.zeros( ( 128 , 128 , 3  ) )
    #Based on class colors, colors the mask particular ways depending on prediction
    for c in range(256):

            seg_img[:,:,0] += ( (pr[:,: ] == c )*( colors[c][0] )).astype('uint8')
            seg_img[:,:,1] += ((pr[:,: ] == c )*( colors[c][1] )).astype('uint8')
            seg_img[:,:,2] += ((pr[:,: ] == c )*( colors[c][2] )).astype('uint8')
    #Resizes mask to original input size
    output = cv2.resize(seg_img  , (image.shape[1] , image.shape[0] ))
    #Converts mask to black/white coloring
    grayImage = cv2.cvtColor(output.astype("float32"), cv2.COLOR_BGR2GRAY)
    (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
    mask=255-blackAndWhiteImage
    mask=cv2.cvtColor(mask,cv2.COLOR_BGR2RGB)
    

    return(mask)


# construct the argument parser and parse the arguments

"""
origimage = cv2.imread("C:/users
documents/example.jpg",1)
origimage=cv2.resize(origimage,(720,540))
image=origimage.copy()
orig = image.copy()
 
# convert the image to grayscale, blur it, and find edges
# in the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)
 
# show the original image and the edge detected image
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
 
# loop over the contours
for c in cnts:
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
 
	# if our approximated contour has four points, then we
	# can assume that we have found our screen
	if len(approx) == 4:
		screenCnt = approx
		break
 
# show the contour (outline) of the piece of paper
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
screenCntArc=cv2.arcLength(screenCnt,True)
"""
def run(filename):
    origimage = cv2.imread(filename,1)
    origimage=cv2.resize(origimage,(720,540))
    image=origimage.copy()
    orig = image.copy()
    card=make_prediction(origimage,model2)
    global ix,iy,drawing,mode
    drawing = False # true if mouse is pressed
    mode = False # if True, draw rectangle. Press 'm' to toggle to curve
    ix,iy = -1,-1
    pred=origimage.copy()
    # mouse callback function
    def draw_circle(event,x,y,flags,param):
        global ix,iy,drawing,mode

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix,iy = x,y

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing == True:
                if mode == True:
                    cv2.rectangle(pred,(ix,iy),(x,y),(0,255,0),-1)
                else:
                    cv2.circle(pred,(x,y),2,255,-1)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            if mode == True:
                cv2.rectangle(pred,(ix,iy),(x,y),(0,255,0),-1)
            else:
                cv2.circle(pred,(x,y),2,255,-1)

    cv2.namedWindow('image')
    cv2.setMouseCallback('image',draw_circle)

    while(1):
        cv2.imshow('image',pred)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('m'):
            mode = not mode
        elif k == 27:
            break

    cv2.destroyAllWindows()

    pred=pred.astype('uint8')
    #gray = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(pred, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)
    
    # show the original image and the edge detected image

    cnts2 = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts2 = imutils.grab_contours(cnts2)
    cnts2 = sorted(cnts2, key = cv2.contourArea, reverse = True)[:5]
    mainCnt=cnts2[0]
    card=card.astype('uint8')
    gray2 = cv2.cvtColor(card, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.GaussianBlur(gray2, (5, 5), 0)
    edged2 = cv2.Canny(gray2, 75, 200)
    
    # show the original image and the edge detected image

    cnts3 = cv2.findContours(edged2.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts3 = imutils.grab_contours(cnts3)
    cnts3 = sorted(cnts3, key = cv2.contourArea, reverse = True)[:5]
    mainCnt2=cnts3[0]
    #rect=cv2.boundingRect(mainCnt2)
    #x,y,w,h = rect

    #cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)


    #Calculate perimeter
    perimeter_of_wound=cv2.arcLength(mainCnt,True)
    perimeter_of_card=cv2.arcLength(mainCnt2,True)

    epsilon = 0.1*perimeter_of_card

    approx = cv2.approxPolyDP(mainCnt2,epsilon,True)

    actual_perimeter_of_card=279.16
    perimeter_of_card=cv2.arcLength(approx,True)
    ratio_card_to_actual=perimeter_of_card/actual_perimeter_of_card
    real_perimeter_of_wound=perimeter_of_wound/ratio_card_to_actual
    bounding_wound_rect=cv2.minAreaRect(mainCnt)
    bounding_wound = cv2.boxPoints(bounding_wound_rect)
    bounding_wound= np.int0(bounding_wound)
    card_width=85.60
    card_height=53.98
    from math import sqrt
    both_width=sqrt((approx[1][0][0]-approx[0][0][0]) **2 + (approx[1][0][1]-approx[0][0][1]) **2) + sqrt((approx[3][0][0]-approx[2][0][0]) **2 + (approx[3][0][1]-approx[2][0][1]) **2)
    dim1=both_width/2
    both_height=sqrt((approx[3][0][0]-approx[0][0][0]) **2 + (approx[3][0][1]-approx[0][0][1]) **2) + sqrt((approx[2][0][0]-approx[1][0][0]) **2 + (approx[2][0][1]-approx[1][0][1]) **2)
    dim2=both_height/2
    if dim2 > dim1:
        pixel_card_width=dim2
        pixel_card_height=dim1
    elif dim1 > dim2:
        pixel_card_width=dim1
        pixel_card_height=dim2

    wounddim1=bounding_wound_rect[1][1]
    wounddim2=bounding_wound_rect[1][0]
    if wounddim2 > wounddim1:
        pixel_wound_width=wounddim2
        pixel_wound_height=wounddim1
    elif wounddim1 > wounddim2:
        pixel_wound_width=wounddim1
        pixel_wound_height=wounddim2
    ratio_card_height=card_height/pixel_card_height
    ratio_card_width=card_width/pixel_card_width
    wound_height=ratio_card_height*pixel_wound_height
    wound_width=ratio_card_width*pixel_wound_width
    print("Approximated perimeter is: " + str(real_perimeter_of_wound))
    print("Approximated width of wound is: " + str(wound_width))
    print("Approximated height of wound is: " +str(wound_height))
    pts1 = np.float32([[0, 720], [0, 0], [720, 0], [720, 540]])
    pts2=np.float32(bounding_wound)
    matrix = cv2.getPerspectiveTransform(pts2, pts1)
    result = cv2.warpPerspective(image, matrix, (720, 540))
    cv2.imwrite("/users/suhas/documents/perspective.jpg",result)

    #print("Perimeter is: "+ str(real_perimeter_of_wound))
    cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)

    cv2.drawContours(image, [mainCnt], -1, (0, 255, 0), 2)
    #cv2.drawContours(image, [mainCnt2], -1, (0, 255, 0), 2)

    #cv2.drawContours(image, [bounding_card], -1, (0, 255, 0), 2)
    cv2.drawContours(image, [bounding_wound], -1, (0, 255, 0), 2)

    cv2.imwrite("/users/suhas/documents/example2.jpg",image)
