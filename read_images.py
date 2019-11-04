# =============================================================================
# Read images 
# =============================================================================
import cv2 as cv
import pickle
import os.path
import numpy as np

#Read the HR imahes 2048x2048
read_path = "D:/Semester_2/Computer_Vision/CV_Project/CV-Project/Data/DIV2K_train_HR/DIV2K_train_HR/"
#Save path for 400x400 images
save_path_hr = "D:/Semester_2/Computer_Vision/CV_Project/CV-Project/Data/train_hr_images/"
#save path for 100x100 images
save_path_lr = "D:/Semester_2/Computer_Vision/CV_Project/CV-Project/Data/train_lr_images/"

#Imageaugmentation function
def random_crop(read_path,save_path,width,height):
    counter=1
    for filename in os.listdir(read_path):
        if filename.endswith(".png"):
            img = cv.imread(read_path+filename)            
            for i in range(8):
                x = np.random.randint(0,img.shape[0]-height)
                y = np.random.randint(0,img.shape[1]-width)
                new_img = img[x:x+height,y:y+width]
                aug_ran = np.random.randint(8)
                #Vertical flip
                if  aug_ran == 1:
                    new_img = cv.flip(new_img, 1)
                #Horizontal flip
                if  aug_ran == 2:
                    new_img = cv.flip(new_img, 0)
                #Add gaussian noise
                if aug_ran == 3:
                    noise = np.zeros(new_img.shape)
                    sig = np.random.randint(4)
                    noise = cv.randn(noise,(0,0,0),(sig,sig,sig))
                    new_img += noise.astype(new_img.dtype)
                    new_img = np.clip(new_img, 0, 255)
                    new_img = np.uint8(new_img)

                #Add brightness
                if aug_ran == 4:
                    brightness = np.uint8(np.random.randint(-10,10),dtype=np.int8)
                    new_img += brightness
                    new_img += brightness
                    new_img = np.clip(new_img, 0, 255)   
                    new_img = np.uint8(new_img)                
                cv.imwrite(save_path + str(counter).zfill(5)+'.png',new_img)
                counter+=1
                
#Generate low resolution images  for the 400x400               
def generate_lr(read_path,save_path,factor):
    for filename in os.listdir(read_path):
        if filename.endswith(".png"):
            img = cv.imread(read_path+filename)
            new_img = cv.resize(img,(img.shape[0]//factor,img.shape[1]//factor),interpolation=cv.INTER_CUBIC)
            cv.imwrite(save_path + filename,new_img)
#train images                
random_crop(read_path,save_path_hr,400,400)
generate_lr(save_path_hr,save_path_lr,4)

#Sampling radom images for validation and test
path_hr = "D:/Semester_2/Computer_Vision/CV_Project/CV-Project/Data/train_hr_images/"
path_lr = "D:/Semester_2/Computer_Vision/CV_Project/CV-Project/Data/train_lr_images/"

val_path_hr = "D:/Semester_2/Computer_Vision/CV_Project/CV-Project/Data/test_hr_images/"
val_path_lr = "D:/Semester_2/Computer_Vision/CV_Project/CV-Project/Data/test_lr_images/"

idx = os.listdir(path_hr)
idx = np.random.choice(idx, 1000, replace=False).tolist() 

#Move files to validation
[os.rename(os.path.join(path_hr,i),os.path.join(val_path_hr,i)) for i in idx]


#Convert images to array and pickle them
#Real and store images in a list

train_lr_path = '/content/train_lr_images/'
train_hr_path = '/content/train_hr_images/'
val_lr_path = '/content/valid_lr_images/'
val_hr_path = '/content/valid_hr_images/'
test_lr_path = '/content/test_lr_images/'
test_hr_path = '/content/test_hr_images/'

def image_data(path):
  return [ cv.imread(path+i) for i in  os.listdir(path)]

train_lr = image_data(train_lr_path)
train_hr = image_data(train_hr_path)

val_lr = image_data(val_lr_path)
val_hr = image_data(val_hr_path)

test_lr = image_data(test_lr_path)
test_hr = image_data(test_hr_path)

#Dumping the data into pickle files
path =''
for i in ['train_lr','train_hr','val_lr','val_hr','test_lr','test_hr']:
  output = open(path+ i +'.pkl', 'wb')
  pickle.dump(eval(i),output)


            
        
        
        