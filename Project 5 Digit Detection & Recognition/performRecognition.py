# Import the modules
import cv2
import numpy as np

# Read the input image 
im = cv2.imread("photo_1.jpg")

# Convert to grayscale and apply Gaussian filtering
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

# Threshold the image
ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)

# Find contours in the image
_,ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get rectangles contains each contour
rects = [cv2.boundingRect(ctr) for ctr in ctrs]

# Loading Models and Weights from JSON and H5 file respectively
from keras.models import model_from_json
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")

from keras.preprocessing.image import img_to_array

for rect in rects:
    # Draw the rectangles
    cv2.rectangle(im, (rect[0]-10, rect[1]-10), (rect[0] + rect[2]+10, rect[1] + rect[3]+10), (0, 255, 0), 3) 
    # Make the rectangular region around the digit
    leng = int(rect[3] * 1.2)
    pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
    pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
    roi = im_th[pt1:pt1+leng , pt2:pt2+leng ]
    # Resize the image
    if roi.any():
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(roi, (3, 3))
        X = img_to_array(roi)
        X = X.reshape(-1,28,28,1)
        nbr = loaded_model.predict(X)
        _,nbr = np.where(nbr == np.amax(nbr))
        cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

#Showing the output    
cv2.imshow("Resulting Image with Rectangular ROIs", im)
cv2.waitKey()

#Saving the output
cv2.destroyAllWindows()
cv2.imwrite("output_2.jpeg",im)