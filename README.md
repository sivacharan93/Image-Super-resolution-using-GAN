##PROJECT NAME
--------------
##IMAGE SUPER-RESOLUTION USING GENERATIVE ADVERSARIAL NETWORKS

##INTRODUCTION
--------------
This project is an implementation of the paper "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network" popularly known as Super-Resolution GAN (SRGAN) \
This project takes input of size 100x100 and upscales it to 400x400 for the Super-resolution task. \
The generator weights are stored after every 5 epochs. \

##REQUIREMENTS
--------------
The following modules are required for the project to run: \
Tensorflow \
Keras \
OpenCV \
Numpy \
Os \
Sys \
Pickle \
Time \

##MODULES
---------
There are four .py files: \

+read_image.py: \
This module takes input as the folder path and reads all the images. \
The images should be high resolution images of size 2000x2000 and the module does augmentation and take 400x400 crop and downsamples it to 100x100 using bicubic interpolation.\

+srgan.py:\
This model reads Numpy arrays of images from the pickle dump from previous module.\
The batch size for training has been set to 8.\
The number of epochs to train for training has been set to 60.\

+inference.py:\
This module takes input image for SR as the file path. \
It should be in the same folder as 'generator.h5' which is the saved model file to load weights of the model.\
It writes image in the same folder with string 'recons' prepended to original image filename.\

+metrics.py:\
This module contains some utility code to do Bilinear, Bicubic and Nearest interpolation for upscaling tasks and PSNR and SSIM value calculation for two arrays of images.\

##CONFIGURATION 
---------------
The model was run on Google Cloud instance with the following configuration:\
8 CPU\
32 GB RAM\
Nvidia Tesla P100 GPU\
100 Gb SSD\

One epoch takes about 1400 seconds on the Google Cloud Instance for 4400 training images of size 100x100x3\

The instructions to set up the instance can be found here:\

1. Follow this selecting the configuration mentioned above:\
https://www.margo-group.com/en/news/running-jupyter-notebook-on-google-cloud-for-a-kaggle-challenge/ \
Download Google Cloud SDK kit for your OS\
While creating the Firewall rule, make it AVAILABLE TO ALL INSTANCES instead of SPECIFIED TAGS\
Prepend Anaconda Path to default path for Python or else "export PATH=~/anaconda3/bin:$PATH" everytime you start the instance\

2. Follow Step2 of this post TILL nvidia-smi\
This is for downloading the GPU drivers into the instance\
https://medium.com/@naomi.fridman/install-conda-tensorflow-gpu-and-keras-on-ubuntu-18-04-1b403e740e25\

You have to do sudo apt-get install ubuntu-drivers-common before checking for the devices.\

##AUTHORS
---------
###SATYARAJA DASARA          sdasara@iu.edu
###SIVA CHARAN MANGAVALLI    simang@iu.edu    
###ADITHYA BOPPANA           aboppana@iu.edu



