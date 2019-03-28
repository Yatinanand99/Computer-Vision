# Project 5 Digit Detection & Recognition

Input Images : Images containing Hand Written Digits with black font and white background.(photo_1,photo_2,photo_3)

Output Images : Images labled with corresponding digits recognized.(output_1,output_2,output_3)

Classifier : Convolutional Neural Network with MNIST Handwritten Digits Dataset

Optimizer : Adam

# Project 6 OCR (Optical Character Recognition)

Input Images : 50,000+ Images of Computerized font and Hand Written Character and Digits. Both Capital and Small Alphabets, 62 Classes.

Output : Trained Model that can classify among 62 Classes of Capital and Small Alphabets and Digits.

Accuracy : 86%

Optimizer : Adam

Link to Dataset : http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/

The Data needed is EnglishHnd.tgz for Hand Written, EnglishFnt.tgz for Computerized Fonts.

The Data is then merged according to the need.

# Project 7 Car Number Plate Detection and Alphanumberic Recognition

Input Images : Images of random Cars Front or Back with Number Plate in it.

Output : Images of corresponding Number Plates with the Recognized Alphanumerics in English.

Processing : 

1. Create edged image of car so that edges can be identified.

2. Find where there is 4 edges with a parameter for good results.

3. Applying various Visual Effects and Finding contours etc for detecting the Alphanumeric.

4. Recognizing the Alphanumerics and labeling them on the image.
