#Import Libraries

import cv2
import numpy as np

#Loading the image
im = cv2.imread("input_3.jpg")

#Converting image to Grayscale
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(im_gray, (5, 5), 0)
edged = cv2.Canny(blurred, 50, 200, 255)

#Finding Contours
import imutils
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
displayCnt = None
 
# loop over the contours
for c in cnts:
	# approximate the contour
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.03 * peri, True)
 
	#The contour has four vertices
	if len(approx) == 4:
		displayCnt = approx
		break

from imutils.perspective import four_point_transform
warped = four_point_transform(im_gray, displayCnt.reshape(4, 2))
output = four_point_transform(im, displayCnt.reshape(4, 2))

thresh = cv2.threshold(warped, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

dim = (720, 240)
# resize image
resized = cv2.resize(thresh, dim, interpolation = cv2.INTER_AREA)

ret, im_th = cv2.threshold(resized, 90, 255, cv2.THRESH_BINARY_INV)

_,ctrs,_ = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

rects = []

for ctr in ctrs: 
    (x, y, w, h) = cv2.boundingRect(ctr)
    if w >=30 and h >= 70:
        rects.append(cv2.boundingRect(ctr))

# Loading Models and Weights from JSON and H5 file respectively
from keras.models import model_from_json
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")

# Load the dictionary of classes for classification labels
label_map = np.load('label_map.npy').item()

from keras.preprocessing.image import img_to_array

#Predicting the alphanumerics
for rect in rects:
    roi = resized[rect[1] - 10:rect[1] + rect[3] + 10, rect[0] - 10:rect[0] + rect[2] + 10]
    if roi.any():
    # Resize the image   
        roi = cv2.resize(roi, (64, 64), interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(roi, (3, 3))
        X = img_to_array(roi)
        X = X/255
        X = X.reshape(-1,64,64,1)
        nbr = loaded_model.predict_classes(X)
        for key in label_map.keys():
            if label_map[key] == nbr:
                nbr =  key
        print (nbr)
        cv2.putText(resized, nbr, (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 2)

#Making the rectangles around the alphanumerics
for rect in rects:
    cv2.rectangle(resized, (rect[0]-10, rect[1]-10), (rect[0] + rect[2]+10, rect[1] + rect[3]+10), (0, 255, 255), 1)

cv2.imshow("Resulting Image with Rectangular ROIs", resized)
cv2.waitKey()

#Saving the output
cv2.destroyAllWindows()
cv2.imwrite("output_3.jpeg",resized)
