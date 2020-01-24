# Python program to illustrate HoughLine
# method for line detection
import cv2
import numpy as np
import math
import subprocess
import pandas as pd

# Reading the required image in
# which operations are to be done.
# Make sure that the image is in the same
# directory in which this python program is

img_sz = 256
centauri = list()

def apply_median_blur(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blured2 = cv2.medianBlur(img_gray, 51)
    divided = np.ma.divide(img_gray, blured2).data
    if divided.max() < 5:
        normed = np.uint8(255 * divided / divided.max())
    else:
        normed = img_gray
    th, threshed = cv2.threshold(normed, 100, 255, cv2.THRESH_OTSU)
    out = cv2.cvtColor(threshed,cv2.COLOR_GRAY2BGR)
    return out

def remove_lines(img):
    cv2.imwrite("input.jpg",img)
    qryStr = "convert input.jpg -type Grayscale -negate -define morphology:compose=darken -morphology\
              Thinning 'Rectangle:1x30+0+0<' -negate out.jpg"
    subprocess.call(qryStr, shell=True)
    return cv2.imread("out.jpg")

def crop_around_center_top_30(img):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point

    Source: http://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    """
    image = (img)
    (width, height) = (image.shape[1], image.shape[0])
    image_center = (int(height * 0.5), int(width * 0.5))
    height_30 = int(0.15*height)
    width_25 = int(0.25*width)
    x2 = int(width_25  + width  * 0.5)

    y2 = int(height_30 + height * 0.5)
    return image[height_30:y2, width_25:x2]

def crop_top_left(img):
    image = (img)
    return image[0:int(image.shape[0]/2),0:int(image.shape[1]/2)]

def crop_top_right(img):
    image = (img)
    return image[ int(image.shape[0]/8) :int(image.shape[0]*(3/8)), int(image.shape[1]/4): int(image.shape[1]*(3/4) ) ]

def crop_around_center(img, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point

    Source: http://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    """
    image = (img)
    image_size = (image.shape[1], image.shape[0])
    #image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))
    image_center = (int(image_size[0] ), int(image_size[1] ))

    # find bigger dim and round img_size to 256,512 or 1024
    #print(image.shape[1], image.shape[0])
    centreScale = [256, 512, 1024, 2048]
    bigDim = min(image.shape[1], image.shape[0])
    width = height = centreScale[np.argmin(
        [abs(centreScale[0] - (0.25) * bigDim), abs(centreScale[1] - (0.25) * bigDim),
         abs(centreScale[2] - (0.25) * bigDim), abs(centreScale[3] - (0.25) * bigDim)])]
    #print('centrepiece = ', width)
    if (width > image_size[0]):
        width = image_size[0]

    if (height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]


CLOCKWISE = 0
COUNTER_CLOCKWISE = 1
def performOtherRot(img, dirn, correctionAngle):
    #print(img)
    fname = (img.split('/')[-1]).split('.')[0]

    qryStr = 'convert -rotate "'+str(correctionAngle)+'" "'+img+'" RES/"'+fname+'_Corrected.jpg"'
    #print( qryStr )
    subprocess.call( qryStr, shell=True )



def performRotation(img, direction, correctionAngle):
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)
    scale = 1.0

    ## since cv2 getRotationMatrix2D works counter clockwise, "direction" arg will remain redundant
    ## but we could always do the necessary math and change def calcCorrectionAngle to return appropriate values
    M = cv2.getRotationMatrix2D(center, correctionAngle, scale)
    correctedImg = cv2.warpAffine(img, M, (h, w))

    ## now calc the mirror/horizontal flip - rotate by  angle = 180
    M = cv2.getRotationMatrix2D(center, 180, scale)
    correctedImg_flip = cv2.warpAffine(correctedImg, M, (h, w))

    return correctedImg, correctedImg_flip


def calcCorrectionAngle(radians):
    theta = math.degrees(radians)
    print('in calcCorrectionAngle - theta is - ', theta)
    if theta >= 0 and theta <= 90:
        return (COUNTER_CLOCKWISE, (90 - theta))
    elif theta > 90 and theta <= 180:
        return (COUNTER_CLOCKWISE, 90 + (180 - theta))
    return None
def cropBasisLargestContour( img, bounder ):
    return img[ bounder[1]: bounder[1]+bounder[3], bounder[0]: bounder[0]+bounder[2] ]

def findBounder( image ):
    import numpy as np
    import cv2 as cv
    img = image
    kernel = np.ones((5,5), np.uint8)
    img_erosion = cv.erode(img, kernel, iterations=1)
    #img_erosion = cv.dilate(img, kernel, iterations=1)
    edges = cv.Canny(img_erosion,100,200)

    cv.imwrite( 'canny.jpg', edges )

    contours,hierarchy=cv.findContours(edges ,cv.RETR_LIST,cv.CHAIN_APPROX_NONE)


    boundRect = [None]*len(contours)
    for i, c in enumerate(contours):
        contours_poly = cv.approxPolyDP(c, 3, True)
        boundRect[i] = cv.boundingRect(contours_poly)

    maxBound =-1
    bounder = None
    prevbounder = None
    for i in range(len(contours)):
        if int( boundRect[i][2] )*int( boundRect[i][3] ) > maxBound:
            maxBound = int( boundRect[i][2] )*int( boundRect[i][3] )
            prevbounder = bounder
            bounder = boundRect[i]
    return bounder
    #return prevbounder

def checkProxima( arr ):
    minDist = 5
    for x1, y1, x2, y2 in centauri:
        if abs( arr[0] - x1 ) < minDist or abs( arr[1] - y1 ) < minDist or abs( arr[2] - x2 ) < minDist or abs( arr[3] - y2 ) < minDist: return True
    return False

def correctOrientation(path):
    orig_image = cv2.imread(path)
    img_blur = apply_median_blur(orig_image)
    kernel = np.ones((3,3), np.uint8)
    #img_blur = cv2.dilate(orig_image, kernel, iterations=1)
    #img_blur = cv2.erode(orig_image, kernel, iterations=1)

    kernel = np.ones((5, 5), np.float32) / 25
    smoothed = cv2.filter2D(orig_image , -1, kernel)

    cv2.imwrite('temp.jpg', crop_around_center_top_30(img_blur))
    #cv2.imwrite('temp.jpg', crop_around_center_top_30( smoothed  ))
    #cv2.imwrite('temp.jpg', crop_top_right( smoothed ))
    #cv2.imwrite( 'temp.jpg',  cropBasisLargestContour( orig_image, findBounder( orig_image )  ) )
    img = cv2.imread('temp.jpg')
    img = remove_lines(img)
    cv2.imwrite("crop_center_inv_.jpg",img)
    # Convert the img to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, im_bw) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # Apply edge detection method on the image
    edges = cv2.Canny(im_bw, 50, 150, apertureSize=3)
    cv2.imwrite('inter.jpg' , edges)
    # This returns an array of r and theta values
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 10)

    # The below for loop runs till r and theta values
    # are in the range of the 2d array

    angles = []
    if len(lines) < 50:
        fin_lines = lines
    else:
        fin_lines = lines[:50]

    prevX1, prevY1, prevX2, prevY2 = 0 , 0 , 0 , 0
    for l in fin_lines:
        for r, theta in l:
            angle = theta
            # Stores the value of cos(theta) in a
            a = np.cos(theta)

            # Stores the value of sin(theta) in b
            b = np.sin(theta)

            # x0 stores the value rcos(theta)
            x0 = a * r

            # y0 stores the value rsin(theta)
            y0 = b * r

            # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
            x1 = int(x0 + 1000 * (-b))

            # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
            y1 = int(y0 + 1000 * (a))

            # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
            x2 = int(x0 - 1000 * (-b))

            # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
            y2 = int(y0 - 1000 * (a))

            # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
            # (0,0,255) denotes the colour of the line to be
            # drawn. In this case, it is red.


            #if checkProxima( [ x1, y1 , x2, y2 ] ):
            #    centauri.append( [ x1, y1 , x2, y2 ] )
            #    continue

            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            centauri.append( [ x1, y1 , x2, y2 ] )

            angles.append(theta)
    #print("total angles", angles)
    cv2.imwrite('intermediate.jpg', img)
    df = pd.Series((v for v in angles))
    freq_map = df.value_counts()
    #print("freq_map",freq_map)
    #print('HIST\n', np.histogram( angles, bins=35, range=(0, 3.5) ) )

    index_max = freq_map.index[0]
    dirn, rotation = calcCorrectionAngle(index_max)
    # return performRotation( orig_image, dirn, rotation )
    return performOtherRot(path, dirn, rotation)


import sys
fname = sys.argv[1]
correctOrientation( fname )
