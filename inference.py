#Importing required libraries
import sys
import os
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
from keras.layers import Input
from keras.models import Model
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.core import Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense
from keras.layers.core import Activation
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras import initializers
from keras.layers import add
from keras import losses
import cv2
import numpy as np

#Defining the model before loading
class Generator(object):

    def __init__(self, img_shape):
        
        self.img_shape = img_shape

    def generator(self):
        
        generator_input = Input(shape = self.img_shape)
        model = Conv2D(filters = 64, kernel_size = 9, strides = 1,kernel_initializer = initializers.he_normal(), padding = "same")(generator_input)
        model = PReLU(alpha_initializer='zeros', shared_axes=[1,2])(model)
	    
        skip_input = model
        
        #Residual blocks
        for index in range(16):
            #Residual block
            #Residual for skip connection
            skip_input_1 = model                
            model = Conv2D(filters = 64, kernel_size = 3, kernel_initializer = initializers.he_normal() ,strides = 1, padding = "same")(model)
            model = BatchNormalization(momentum = 0.6)(model)
            model = PReLU(alpha_initializer='zeros', shared_axes=[1,2])(model)
            model = Conv2D(filters = 64, kernel_size = 3,kernel_initializer = initializers.he_normal(), strides = 1, padding = "same")(model)
            model = BatchNormalization(momentum = 0.6)(model)    
            model = add([skip_input_1, model])   

	    
        model = Conv2D(filters = 64, kernel_size = 3, strides = 1,kernel_initializer = initializers.he_normal(), padding = "same")(model)
        model = BatchNormalization(momentum = 0.6)(model)
        model = add([skip_input, model])
	    
       #Upsampling blocks
        for index in range(2):
            model = Conv2DTranspose( kernel_size = 3, filters = 256, strides = 2,kernel_initializer = initializers.he_normal(), padding = "same")(model)
            model = LeakyReLU(alpha = 0.25)(model)
	    
	    #Final layer to set the channels to 3 of input image
        model = Conv2D(filters = 3, kernel_size = 9, kernel_initializer = initializers.he_normal(),strides = 1, padding = "same")(model)
        	    #Using tanh activation to get the values in the range of  [-1,1]
        model = Activation('tanh')(model)
	   
        return Model(inputs = generator_input, outputs = model)

def normalize(input_data):
    return (np.float32(input_data) - 127.5)/127.5 

def denormalize(input_data):
    return np.uint8(np.int32((np.float32(input_data) * 127.5) + 127.5 ))

#Load the generator model
g = Generator((100,100,3)).generator()
g.load_weights('generator.h5')

#Do the prediction on the image and save
usr_args = sys.argv[1:]
file_name = str(usr_args[0])

inp = cv2.imread(file_name)
inp = normalize(inp)
inp = np.reshape(inp,(1,100,100,3))
inp = g.predict(inp)
inp = np.reshape(inp,(400,400,3))
inp = denormalize(inp)
new_file_name = 'recons_'+file_name
cv2.imwrite(new_file_name,inp)
print('File written as : ',new_file_name)
