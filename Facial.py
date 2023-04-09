#!/usr/bin/env python
# coding: utf-8

# # Project : "Mask" or "Without Mask" 

# # 1.Setting our DataSet:

# In[1]:


images = []
labels = []


# In[2]:


import os


# In[3]:


# Exporting our DataSet provided:

os.listdir("C:/Users/DELL/NEURAL/dataset_mask")


# In[4]:


import cv2


# In[5]:


# Refining done on our Dataset:

# Scanning the folders:

for i in ["without_mask", "with_mask"]:
    imageNameList = os.listdir("C:/Users/DELL/NEURAL/dataset_mask/" + str(i))
    
    # Scanning the Images:
    
    for fileName in imageNameList:
        Image = cv2.imread("C:/Users/DELL/NEURAL/dataset_mask/" + str(i) + "/" + str(fileName))
        try:
            gray = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
            Image = cv2.resize(gray,(100,100))  # resizing the image to (100,100) dimensions
            images.append(Image)
            
            if i == "without_mask":
                labels.append(0)
                
            else:
                labels.append(1)
                
        except:
            pass
        
    print("Inside folder ",i)
    
    
# Error Exception Method is used here to avoid program stopping due to Different Image Sizes.


# # Converting All the images to Array form:

# In[6]:


import numpy as np


# In[7]:


images = np.array(images)
labels = np.array(labels)


# # Some imp Refining for our Model:

# In[8]:


from sklearn.model_selection import train_test_split


# In[9]:


train_features, test_features, train_targets, test_targets = train_test_split(images, labels, test_size = 0.2)


# In[10]:


def preprocessing(img):
    img = img/255
    return img


# # Defining our Training & Testing Features:

# *MAP* method is used to save our computation Time.
# 
# Instead of using loop for each image and then applying *preprocessing* function, we will use this *MAP* method.

# In[11]:


train_features = np.array(list(map(preprocessing, train_features)))


# In[12]:


train_features.shape


# In[13]:


# Reshaping is to be done to add Color channels

train_features = train_features.reshape(train_features.shape[0], train_features.shape[1], train_features.shape[2], 1)


# In[14]:


train_features.shape


# In[15]:


test_features.shape


# In[16]:


test_features=test_features.reshape(test_features.shape[0],test_features.shape[1],test_features.shape[2],1)


# In[17]:


test_features.shape


# # Applying Image Augmentation over our dataset: 
# 
# ---- For Better Convolution NN ----

# In[18]:


from keras.preprocessing.image import ImageDataGenerator


# In[19]:


dataGenerator = ImageDataGenerator(width_shift_range = 0.1, height_shift_range = 0.1, zoom_range = 0.2, shear_range = 0.1, rotation_range = 10)


# In[20]:


# Now applyig these changes to our training Features

dataGenerator.fit(train_features)


# In[21]:


## Something about BATCHES is to be written ##


# In[22]:


batches = dataGenerator.flow(train_features, train_targets, batch_size = 20)


# In[23]:


x_batch, y_batch = next(batches)


# # Displaying Images:

# In[24]:


import matplotlib.pyplot as plt


# In[25]:


plt.figure(figsize=(10,10))

for i in range (20):
    plt.subplot(4, 5, i+1)
    plt.imshow(x_batch[i].reshape(100,100))
    
plt.show()


# # Now, we will Build Our Model:

# Step_1 : INCREASING THE NUMBER OF TARGET COLUMNS....

# In[26]:


train_targets.shape


# In[27]:


from tensorflow.keras.utils import to_categorical


# In[28]:


# Applyig ONE HOT ENCODING to our training targets and Testing targets

train_targets = to_categorical(train_targets)

test_targets  = to_categorical(test_targets)


# In[29]:


train_targets.shape


# In[30]:


test_targets.shape


# ONE HOT ENCODING ----- successfully applied

# In[31]:


from keras.models import Sequential
from keras.layers import Flatten, Conv2D, MaxPooling2D, Dense, Dropout


# # Step_1: Specifying the Architecture:

# In[32]:


model = Sequential()

model.add(Conv2D(80,(5,5), activation="relu", input_shape=(100,100,1)))
model.add(Conv2D(80,(5,5), activation="relu"))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(80,(5,5), activation="relu"))
model.add(Conv2D(50,(5,5), activation="relu"))
model.add(MaxPooling2D((2,2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(500, activation="relu"))
model.add(Dense(2, activation="softmax"))


model.summary()


# # Step_2: Compile the Model:

# In[33]:


from tensorflow.keras.optimizers import Adam


# In[34]:


model.compile(Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])


# # Step_3: Fit the Model:

# In[35]:


train_features.shape


# In[36]:


train_targets.shape


# In[37]:


model.fit(dataGenerator.flow(train_features, train_targets, batch_size=20), epochs=10)


# def getClassName(classNo):
#     if   classNo == '0': return 'without_mask'
#     elif classNo == '1': return 'with_mask' 

# --- For the purpose of Testing Our Model --- 

# In[38]:


# Load the Cascade
face_cascade = cv2.CascadeClassifier("C:/Users/DELL/NEURAL/HaarCascade/haarcascade_frontalface_default.xml")

# Read The input Image (array)
img = cv2.imread("C:/Users/DELL/NEURAL/lena.png")

# Converting into GrayScale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect Faces 
faces = face_cascade.detectMultiScale(gray, 1.1, 5)

# Drawing Rectangles over the faces:
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x,y),(x+w,y+h), (0,255,0), 2)
    face_img = gray[y:y+h, x:x+h]  # Cropping the faces
    
# Displaying the Output:
cv2.imshow('Detections', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[39]:


face_img.shape


# In[40]:


labels_dict={1:'MASK',0:'NO MASK'}
color_dict={1:(0,255,0),0:(0,0,255)}


# In[41]:


video = cv2.VideoCapture(0)


# In[45]:


while(True):

    ret,img=video.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(img,1.3,5)  

    for (x,y,w,h) in faces:
    
        face_img=gray[y:y+w,x:x+w]
        resized=cv2.resize(face_img,(100,100))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,100,100,1))
        result = model.predict(reshaped)

        label=np.argmax(result,axis=1)[0]
        
        cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1)
        cv2.putText(img, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
        
    cv2.imshow('LIVE',img)
    key=cv2.waitKey(1)
    
    if(key==27):  # ESC key
        break
        
cv2.destroyAllWindows()
video.release()


# In[46]:


result


# In[ ]:




