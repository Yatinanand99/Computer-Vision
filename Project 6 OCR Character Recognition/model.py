
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the CNN
classifier = Sequential()

# Convolution
classifier.add(Convolution2D(64, 5, 5, input_shape = (64, 64, 1), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding more convolutional layer
classifier.add(Convolution2D(64, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Convolution2D(128, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Convolution2D(64, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Dropout(0.2))

# Flattening
classifier.add(Flatten())

# Full connection
classifier.add(Dense(output_dim = 256, activation = 'relu'))
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dropout(0.4))
classifier.add(Dense(output_dim = 62, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Augumenting the data for better results in less data
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   featurewise_std_normalization=False,  # divide inputs by std of the dataset
                                   samplewise_std_normalization=False,  # divide each input by its std
                                   zca_whitening=False, 
                                   rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180) 
                                   width_shift_range=0.1, 
                                   height_shift_range=0.1, 
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = False,
                                   validation_split = 0.25)


training_set = train_datagen.flow_from_directory('English\Fnt',color_mode="grayscale",
                                                 target_size = (64, 64),
                                                 batch_size = 64,
                                                 class_mode = 'categorical',
                                                 subset = 'training')

validation_set = train_datagen.flow_from_directory('English\Fnt',color_mode="grayscale",
                                                 target_size = (64, 64),
                                                 batch_size = 64,
                                                 class_mode = 'categorical',
                                                 subset = 'validation')

#Make a Dictionary of classes and labels
label_map = (training_set.class_indices)

import numpy as np
# Save the dictionary
np.save('label_map.npy',label_map) 

# Load the dictionary
label_map = np.load('label_map.npy').item()

# Fitting the classifier to dataset
classifier.fit_generator(training_set,
                              epochs = 20,steps_per_epoch = 2000,
                              verbose = 1,validation_steps = 1000, validation_data = validation_set)

# Saving Models and Weights to a JSON and H5 file respectively
model_json = classifier.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
classifier.save_weights("model.h5")

# Loading Models and Weights from JSON and H5 file respectively
from keras.models import model_from_json
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
