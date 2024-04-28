#!/usr/bin/env python
# coding: utf-8
#IMAGE CLASSIFICATION USING DEEP LEARNING
# In[1]:
import tensorflow as tf  #used to import the Tensorflow library
# In[2]:
import numpy as np   #used to import the numpy library as it supports multi dimensional array

#FIRST IMAGE 

# In[3]:
filename=r"C:\Users\kriti\Downloads\project2.jpeg"
# In[4]:
from tensorflow.keras.preprocessing import image   #method to load the image  (input image is converted into pixels of size 224,224)
img=image.load_img(filename, target_size= (224,224))  #image.load_img loads the image from specific filename
# In[5]:
import matplotlib.pyplot as plt  ##matplotlib is a python plotting library 
# In[6]:
plt.imshow(img)  

#LOAD THE DEEP LEARNING MODEL
# In[7]:
mobile=tf.keras.applications.mobilenet.MobileNet()  #DEEP LEARNING MODEL WEIGHTS- pre-trained

#PRE-PROCESSING OF THE IMAGE

# In[8]:
from tensorflow.keras.preprocessing import image
img=image.load_img(filename, target_size= (224,224))
# In[9]:
plt.imshow(img)
# In[10]:
resized_img=image.img_to_array(img)  #convert to numpy array  # resizing
final_image=np.expand_dims(resized_img,axis=0) #need fourth dimension (batch dimension)
final_image=tf.keras.applications.mobilenet.preprocess_input(final_image)

# In[11]:
final_image.shape   # (batch size,height,width,channels )
# In[12]:
predictions=mobile.predict(final_image)
# In[13]:
from tensorflow.keras.applications import imagenet_utils #provides functions specific to work with models trained on the ImageNet dataset.
# In[14]:
result=imagenet_utils.decode_predictions(predictions) #convert the raw predictions (obtain from neural network model)into human readable format
# In[15]:

print(result) ###  OUTPUT


# #SECOND IMAGE
# In[16]:
filename=r"C:\Users\kriti\Downloads\project1.jpeg"
# In[17]:
from tensorflow.keras.preprocessing import image   #method to load the image
img=image.load_img(filename, target_size= (224,224))
# In[18]:
import matplotlib.pyplot as plt
# In[19]:
plt.imshow(img)
# In[20]:
#load the deep learning model
mobile=tf.keras.applications.mobilenet.MobileNet()
# In[21]:

#pre-processing of the image
from tensorflow.keras.preprocessing import image
img=image.load_img(filename, target_size= (224,224))
# In[22]:
plt.imshow(img)
# In[23]:

resized_img=image.img_to_array(img)
final_image=np.expand_dims(resized_img,axis=0)
final_image=tf.keras.applications.mobilenet.preprocess_input(final_image)

# In[24]:
final_image.shape
# In[25]:

predictions=mobile.predict(final_image)
# In[26]:
from tensorflow.keras.applications import imagenet_utils
# In[27]:
result=imagenet_utils.decode_predictions(predictions)
# In[28]:

print(result) #### OUTPUT


# #THIRD IMAGE
# In[29]:
filename=r"C:\Users\kriti\Downloads\project3.jpg"
# In[30]:

from tensorflow.keras.preprocessing import image #LOAD THE IMAGE
img=image.load_img(filename, target_size= (224,224))
# In[31]:
import matplotlib.pyplot as plt
# In[32]:
plt.imshow(img)
# In[33]:
mobile=tf.keras.applications.mobilenet.MobileNet() #LOAD THE DEEP LEARNING MODEL


# In[34]:
from tensorflow.keras.preprocessing import image  #PRE-PROCESSING OF THE IMAGE
img=image.load_img(filename, target_size= (224,224))
# In[35]:
plt.imshow(img)

# In[36]:
resized_img=image.img_to_array(img)
final_image=np.expand_dims(resized_img,axis=0)
final_image=tf.keras.applications.mobilenet.preprocess_input(final_image)


# In[37]:
final_image.shape


# In[38]:
predictions=mobile.predict(final_image)


# In[39]:
from tensorflow.keras.applications import imagenet_utils


# In[40]:
result=imagenet_utils.decode_predictions(predictions)


# In[41]:
print(result) ### OUTPUT


# In[ ]:




