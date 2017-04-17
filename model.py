import os
import csv
import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle
from keras.layers import Input, Flatten, Dense, Activation, Dropout, Lambda, Cropping2D
from keras.models import Model, Sequential 
from keras.activations import relu, softmax
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import matplotlib.pyplot as plt
from random import randint
from keras.optimizers import Adam
from keras.regularizers import l2

### PARAMETERS
EPOCHS = 20 
DROPOUT_P = 0.3
visualize_input_data = 1
visualize_loss = 1
use_lr_camera = 1
use_flipped = 1
angle_correction = 0.24 
use_generator = 0 
VALIDATION_SPLIT = 0.20
BATCH_SIZE = 128


def normalize_data(data):
    data = np.asarray(data)
    st = data[:,3].astype(np.float32)
    bins, number_per_bin = 100, 120  #100, 150
    hist, bin_edges = np.histogram(st, bins)
    indices = np.digitize(st, bin_edges)
    samples = np.concatenate([data[indices==x][:number_per_bin] for x in range(bins)])
    return samples


def plotdata(y_train, name):
    input_angles = y_train
    #np.array([float(x[3]) for x in samples])
    print(len(input_angles))
    plt.hist(input_angles, bins=50)
    plt.title("Distribution of Angle Data")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()
    #plt.savefig(name)

samples = []
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        samples.append(line)

if(visualize_input_data):
    plotdata(np.array([float(x[3]) for x in samples]), 'angles_original.png')

samples = normalize_data(samples)

if(visualize_input_data):
    plotdata(np.array([float(x[3]) for x in samples]), 'angles_normalized.png')

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=VALIDATION_SPLIT)


def process_image(name):
    image = cv2.imread(name)
    return image

#batch size is x, but x*3 per sample are generated for (left, center, right)
def generator(samples, batch_size=BATCH_SIZE):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './IMG/'+batch_sample[0].split('/')[-1]
                #add center image & angle
                center_image = process_image(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                if(use_flipped):  
                    #append flipped center images
                    images.append(np.fliplr(center_image))
                    angles.append(-center_angle)  
                if(use_lr_camera):
                    left_angle = center_angle + angle_correction
                    right_angle = center_angle - angle_correction
                    #add left image & angle
                    name = './IMG/'+batch_sample[1].split('/')[-1]
                    left_image = process_image(name)
                    images.append(left_image)
                    angles.append(left_angle)  
                    #add right image & angle
                    name = './IMG/'+batch_sample[2].split('/')[-1]
                    right_image = process_image(name)
                    images.append(right_image)
                    angles.append(right_angle)  

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


def getFullDataSet(samples):
            num_samples = len(samples)
            images = []
            angles = []
            for batch_sample in samples:
                name = './IMG/'+batch_sample[0].split('/')[-1]
                #add center image & angle
                center_image = process_image(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                if(use_flipped):  
                    #append flipped center images
                    images.append(np.fliplr(center_image))
                    angles.append(-center_angle)  
                if(use_lr_camera):
                    left_angle = center_angle + angle_correction
                    right_angle = center_angle - angle_correction
                    #add left image & angle
                    name = './IMG/'+batch_sample[1].split('/')[-1]
                    left_image = process_image(name)
                    images.append(left_image)
                    angles.append(left_angle)   
                    #add right image & angle
                    name = './IMG/'+batch_sample[2].split('/')[-1]
                    right_image = process_image(name)
                    images.append(right_image)
                    angles.append(right_angle)  

            shuffle(images, angles)
            X_train = np.array(images)
            y_train = np.array(angles)

            if(visualize_input_data):
                plotdata(y_train, 'angles_training.png')
            return sklearn.utils.shuffle(X_train, y_train)


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)


def model():
    ch, row, col = 3, 80, 320  # Trimmed image format

    model = Sequential()
    #Crop out the sky and other not relevant areas
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
    # Preprocess incoming data, centered around zero with small standard deviation 
    model.add(Lambda(lambda x: x/127.5 - 1.))

    #Model based on NVIDIA architecure
    #  http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
    model.add(Convolution2D(24, 5, 5, activation='elu', subsample=(2,2)))
    model.add(Dropout(DROPOUT_P))
    model.add(Convolution2D(36, 5, 5, activation='elu', subsample=(2,2)))
    model.add(Dropout(DROPOUT_P))
    model.add(Convolution2D(48, 5, 5, activation='elu', subsample=(2,2)))
    model.add(Dropout(DROPOUT_P))
    model.add(Convolution2D(64, 3, 3, activation='elu'))
    model.add(Dropout(DROPOUT_P))
    model.add(Convolution2D(64, 3, 3, activation='elu'))
    model.add(Dropout(DROPOUT_P))
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Flatten())
    model.add(Dense(1164, activation='elu'))
    model.add(Dropout(DROPOUT_P))
    model.add(Dense(100, activation='elu'))
    model.add(Dropout(DROPOUT_P))
    model.add(Dense(50, activation='elu'))
    model.add(Dropout(DROPOUT_P))
    model.add(Dense(10, activation='elu'))
    model.add(Dropout(DROPOUT_P))
    model.add(Dense(10, activation='elu'))
    model.add(Dropout(DROPOUT_P))
    model.add(Dense(1, activation='linear'))


    model.compile(loss='mse', optimizer=Adam())
    return model

model = model()

from keras.utils.visualize_util import plot
plot(model, to_file='model.png', show_shapes=1)

if(use_generator):
    samples_per_epoch = len(train_samples) + use_lr_camera * 2 * len(train_samples) + use_flipped * len(train_samples) 
    nb_val_samples = len(validation_samples) + use_lr_camera * 2 * len(validation_samples) + use_flipped * len(validation_samples)

    history = model.fit_generator(generator=train_generator, 
                        samples_per_epoch=samples_per_epoch, 
                        validation_data=validation_generator,
                        nb_val_samples=nb_val_samples, 
                        nb_epoch=EPOCHS, verbose=1)
else:
    X_train, y_train = getFullDataSet(samples)
    history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=EPOCHS, verbose=1, validation_split=VALIDATION_SPLIT)

model.save('model.h5')
#model.summary()


### plot the training and validation loss for each epoch
if(visualize_loss):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()
    #plt.savefig('loss.png')
