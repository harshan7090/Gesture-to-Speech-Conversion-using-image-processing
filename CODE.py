#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# IMPORTING LIBRARIES 

import cv2
import numpy as np
import os

# CREATING NEW DIRECTORIES 

if not os.path.exists("mydata"):
    os.makedirs("mydata")
    os.makedirs("mydata/train")
    os.makedirs("mydata/test")
    os.makedirs("mydata/train/Nothing")
    os.makedirs("mydata/train/Hello")
    os.makedirs("mydata/train/Yes")
    os.makedirs("mydata/train/No")
    os.makedirs("mydata/train/ThankYou")
    os.makedirs("mydata/train/Help")
    os.makedirs("mydata/train/You")
    os.makedirs("mydata/train/Wow")
    os.makedirs("mydata/train/What")
    os.makedirs("mydata/train/Emergency")
    os.makedirs("mydata/train/Done")
    os.makedirs("mydata/train/Iloveyou")
    os.makedirs("mydata/train/Play")
    os.makedirs("mydata/train/Coffee")
    os.makedirs("mydata/train/Peace")
    os.makedirs("mydata/train/Climb")
    os.makedirs("mydata/train/Dislike")
    os.makedirs("mydata/train/Liedown")
    os.makedirs("mydata/train/More")
    os.makedirs("mydata/train/Cold")
    os.makedirs("mydata/test/Nothing")
    os.makedirs("mydata/test/Hello")
    os.makedirs("mydata/test/Yes")
    os.makedirs("mydata/test/No")
    os.makedirs("mydata/test/ThankYou")
    os.makedirs("mydata/test/Help")
    os.makedirs("mydata/test/You")
    os.makedirs("mydata/test/Wow")
    os.makedirs("mydata/test/What")
    os.makedirs("mydata/test/Emergency")
    os.makedirs("mydata/test/Done")
    os.makedirs("mydata/test/Iloveyou")
    os.makedirs("mydata/test/Play")
    os.makedirs("mydata/test/Coffee")
    os.makedirs("mydata/test/Peace")
    os.makedirs("mydata/test/Climb")
    os.makedirs("mydata/test/Dislike")
    os.makedirs("mydata/test/Liedown")
    os.makedirs("mydata/test/More")
    os.makedirs("mydata/test/Cold")
    
# TRAIN OR TEST

mode = 'train'
directory = 'mydata/'+mode+'/'
cap = cv2.VideoCapture(0)
while True:
        frame = cap.read()
    
# SIMULATING MIRROR IMAGE 

frame = cv2.flip(frame, 1)

# TO GET COUNT OF EXISTING IMAGES IN THE DIRECTORY

sign = {'Nothing' : len(os.listdir(directory+"/Nothing")),
    'Hello': len(os.listdir(directory+"/Hello")),
    'Yes': len(os.listdir(directory+"/Yes")),
    'No' : len(os.listdir(directory+"/No")),
    'ThankYou': len(os.listdir(directory+"/ThankYou")),
    'Help' : len(os.listdir(directory+"/Help")),
    'You' : len(os.listdir(directory+"/You")),
    'Wow' : len(os.listdir(directory+"/Wow")),
    'What' : len(os.listdir(directory+"/What")),
    'Emergency' : len(os.listdir(directory+"/Emergency")),
    'Done' : len(os.listdir(directory+"/Done")),
    'Iloveyou' : len(os.listdir(directory+"/Iloveyou")),
    'Play' : len(os.listdir(directory+"/Play")),
    'Coffee' : len(os.listdir(directory+"/Coffee")),
    'Peace' : len(os.listdir(directory+"/Peace")),
    'Climb' : len(os.listdir(directory+"/Climb")),
    'Dislike' : len(os.listdir(directory+"/Dislike")),
    'Liedown' : len(os.listdir(directory+"/Liedown")),
    'More' : len(os.listdir(directory+"/More")),
    'Cold' : len(os.listdir(directory+"/Cold"))
    }

# TO ADD TEXT ON IMAGE IN THE SCREEN

cv2.putText(frame, "MODE : "+mode, (10, 50), 
cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
cv2.putText(frame, "IMAGE COUNT", (10, 80),
cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
cv2.putText(frame,"NOTHING:"+str(sign['Nothing']),(10,100),
cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
cv2.putText(frame,"HELLO:"+str(sign['Hello']),(10,120),
cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
cv2.putText(frame,"YES:"+str(sign['Yes']),(10,140),
cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
cv2.putText(frame,"NO:"+str(sign['No']),(10,160),
cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
cv2.putText(frame,"THANKYOU:"+str(sign['ThankYou']),(10,180),
cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
cv2.putText(frame,"HELP:"+str(sign['Help']),(10,200),
cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
cv2.putText(frame,"YOU:"+str(sign['You']),(10,220),
cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
cv2.putText(frame,"WOW:"+str(sign['Wow']),(10,240),
cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
cv2.putText(frame,"WHAT:"+str(sign['What']),(10,260),
cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
cv2.putText(frame,"EMERGENCY:"+str(sign['Emergency']),(10,280),
cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
cv2.putText(frame,"DONE:"+str(sign['Done']),(10,300),
cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
cv2.putText(frame,"ILOVEYOU:"+str(sign['Iloveyou']),(10,320),
cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
cv2.putText(frame,"PLAY:"+str(sign['Play']),(10,340),
cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
cv2.putText(frame,"COFFEE:"+str(sign['Coffee']),(10,360),
cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
cv2.putText(frame,"PEACE:"+str(sign['Peace']),(10,380),
cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
cv2.putText(frame,"CLIMB:"+str(sign['Climb']),(10,400),
cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
cv2.putText(frame,"DISLIKE:"+str(sign['Dislike']),(10,420),
cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
cv2.putText(frame,"LIEDOWN:"+str(sign['Liedown']),(10,440),
cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
cv2.putText(frame,"MORE:"+str(sign['More']),(10,460),
cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
cv2.putText(frame,"COLD:"+str(sign['Cold']),(10,480),
cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)

# COORDINATES OF ROI (REGION OF INTEREST)

x1 = int(0.5*frame.shape[1])
y1 = 10
x2 = frame.shape[1]-10
37
y2 = int(0.5*frame.shape[1])

# TO DRAW THE ROI (HERE WE ARE USING RECTANGLE BOX TO MARK THE ROI)
# THE INCREMENT/DECREMENT BY 1 IS TO COMPENSATE FOR THE BOUNDING BOX

cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)

# EXTRACTING ROI FROM THE IMAGE 

roi = frame[y1:y2, x1:x2]
roi = cv2.resize(roi, (64, 64))

# PROCESSING THE CAPTURED IMAGE

cv2.imshow("Frame", frame)
roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
roi = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
cv2.imshow("ROI", roi)
interrupt = cv2.waitKey(10)
if interrupt & 0xFF == 27: # esc key
    break
    if interrupt & 0xFF == ord('0'):
        cv2.imwrite(directory+'Nothing/'+str(sign['Nothing'])+'.jpg', roi)
    if interrupt & 0xFF == ord('1'):
        cv2.imwrite(directory+'Hello/'+str(sign['Hello'])+'.jpg', roi)
    if interrupt & 0xFF == ord('2'):
        cv2.imwrite(directory+'Yes/'+str(sign['Yes'])+'.jpg', roi)
    if interrupt & 0xFF == ord('3'):
        cv2.imwrite(directory+'No/'+str(sign['No'])+'.jpg', roi)
    if interrupt & 0xFF == ord('4'):
        cv2.imwrite(directory+'ThankYou/'+str(sign['ThankYou'])+'.jpg', roi)
    if interrupt & 0xFF == ord('5'):
        cv2.imwrite(directory+'Help/'+str(sign['Help'])+'.jpg', roi)
    if interrupt & 0xFF == ord('6'):
        cv2.imwrite(directory+'You/'+str(sign['You'])+'.jpg', roi)
    if interrupt & 0xFF == ord('7'):
        cv2.imwrite(directory+'Wow/'+str(sign['Wow'])+'.jpg', roi)
    if interrupt & 0xFF == ord('8'):
        cv2.imwrite(directory+'What/'+str(sign['What'])+'.jpg', roi)
    if interrupt & 0xFF == ord('9'):
        cv2.imwrite(directory+'Emergency/'+str(sign['Emergency'])+'.jpg', roi)
    if interrupt & 0xFF == ord('a'):
        cv2.imwrite(directory+'Done/'+str(sign['Done'])+'.jpg', roi)
    if interrupt & 0xFF == ord('b'):
        cv2.imwrite(directory+'Iloveyou/'+str(sign['Iloveyou'])+'.jpg', roi)
    if interrupt & 0xFF == ord('c'):
        cv2.imwrite(directory+'Play/'+str(sign['Play'])+'.jpg', roi)
    if interrupt & 0xFF == ord('d'):
        cv2.imwrite(directory+'Coffee/'+str(sign['Coffee'])+'.jpg', roi)
    if interrupt & 0xFF == ord('e'):
        cv2.imwrite(directory+'Peace/'+str(sign['Peace'])+'.jpg', roi)
    if interrupt & 0xFF == ord('f'):
        cv2.imwrite(directory+'Climb/'+str(sign['Climb'])+'.jpg', roi)
    if interrupt & 0xFF == ord('g'):
        cv2.imwrite(directory+'Dislike/'+str(sign['Dislike'])+'.jpg', roi)
    if interrupt & 0xFF == ord('h'):
        cv2.imwrite(directory+'Liedown/'+str(sign['Liedown'])+'.jpg', roi)
    if interrupt & 0xFF == ord('i'):
        cv2.imwrite(directory+'More/'+str(sign['More'])+'.jpg', roi)
    if interrupt & 0xFF == ord('j'):
        cv2.imwrite(directory+'Cold/'+str(sign['Cold'])+'.jpg', roi)

cap.release()

cv2.destroyAllWindows()

# TRAINING DATASET

# IMPORTING THE KERAS LIBRARIES AND PACKAGES

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
import matplotlib.pyplot as plt

# STEP - 1 BUILDING THE CNN
# INITIALIZING THE CNN

classifier = Sequential()

# FIRST CONVOLUTION LAYER AND POOLING

classifier.add(Convolution2D(32, (3, 3), input_shape=(64, 64, 1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# SECOND CONVOLUTION LAYER AND POOLING 

classifier.add(Convolution2D(32, (3, 3), activation='relu'))

# INPUT_SHAPE IS GOING TO BE THE POOLED FEATURE MAPS FROM THE PREVIOUS CONVOLUTION LAYER 

classifier.add(MaxPooling2D(pool_size=(2, 2)))

# THIRD CONVOLUTION LAYER AND POLLING

classifier.add(Convolution2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# FOURTH CONVOLUTION LAYER AND POOLING

classifier.add(Convolution2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# FLATTENING THE LAYERS

classifier.add(Flatten())

# ADDING A FULLY CONNECTED LAYER

classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units=20, activation='softmax'))    # softmax for more than 2

# COMPILING THE CNN

classifier.compile(optimizer='adam',loss='categorical_crossentropy',
metrics=['accuracy'])    # categorical_crossentropy for more than 2

# STEP 2 - PREPARING THE TRAIN / TEST DATA AND TRAINING MODEL

# CODE COPIED FROM - https://keras.io/preprocessing/image/

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory('mydata/train',
                                target_size=(64, 64),
                                batch_size=1,
                                color_mode='grayscale',
                                class_mode='categorical')
test_set = test_datagen.flow_from_directory('mydata/test',
                            target_size=(64, 64),
                            batch_size=1,
                            color_mode='grayscale',
                            class_mode='categorical')
print(classifier.summary())
history =classifier.fit(
    training_set,
    steps_per_epoch= 6011,   # No of images in training set
    epochs=50,
    validation_data=test_set,
    validation_steps=2102)   # No of images in test set
test_eval = classifier.evaluate(test_set, verbose=0)
print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])

# LOSS PLOT 

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()

# ACCURACY PLOT 

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()

# SAVING THE MODEL 

model_json = classifier.to_json()
with open("oursecmodel-bw.json", "w") as json_file:
    json_file.write(model_json)
classifier.save_weights('oursecmodel-bw.h5')

# PREDICTION 

import numpy as np
from keras.models import model_from_json
import operator
import cv2
import sys, os
from gtts import gTTS

# LOADING THE MODEL 

json_file = open("ourmodel-bw.json", "r")
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)

# LOAD WEIGHTS INTO THE NEW MODEL

loaded_model.load_weights("ourmodel-bw.h5")
print("Loaded model from disk")
cap = cv2.VideoCapture(0)

# CATEGORY DICTIONARY 

categories = {0: 'Nothing', 1: 'Hello', 2: 'Yes', 3: 'No', 4: 'ThankYou', 5: 'Help', 6: 'You', 
              7: 'Wow', 8: 'What', 9: 'Emergency', 10: 'Done',11: 'Climb',12: 'PLay',13: 'Liedown',
              14: 'More',15: 'Cold',16: 'Iloveyou',17: 'Peace',18: 'Dislike',19: 'Coffee'
             }
while True:
    _, frame = cap.read()
    
# SIMULATING MIRROR IMAGE 

frame = cv2.flip(frame, 1)

# GOT THIS FROM collect-data.py

# COORDINATES OF THE ROI

x1 = int(0.5*frame.shape[1])
y1 = 10
x2 = frame.shape[1]-10
y2 = int(0.5*frame.shape[1])

# TO DRAW THE ROI (HERE WE ARE USING RECTANGLE BOX TO MARK THE ROI)
# THE INCREMENT/DECREMENT BY 1 IS TO COMPENSATE FOR THE BOUNDING BOX

cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (0,255,0) ,1)

# EXTRACTING THE ROI 

roi = frame[y1:y2, x1:x2]

# RESIZING THE ROI SO IT CAN BE FED TO THE MODEL FOR PREDICTION 

roi = cv2.resize(roi, (64, 64))
roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
test_image = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)
cv2.imshow("test", test_image) # Batch of 1
result = loaded_model.predict(test_image.reshape(1, 64, 64, 1))
prediction = {'Nothing' : result[0][12],
              'Hello': result[0][6],
              'Yes': result[0][18],
              'No': result[0][11],
              'ThankYou': result[0][15],
              'Help': result[0][7],
              'You': result[0][19],
              'Wow': result[0][17],
              'What': result[0][16],
              'Emergency': result[0][5],
              'Done': result[0][4],
              'Climb': result[0][0],
              'Play': result[0][14],
              'Liedown': result[0][9],
              'More': result[0][10],
              'Cold': result[0][2],
              'Iloveyou': result[0][8],
              'Peace': result[0][13],
              'Dislike': result[0][3],
              'Coffee': result[0][1]
             }

# SORTING BASED ON TOP PREDICTION 

prediction=sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)

# DISPLAYING THE PREDICTIONS

cv2.putText(frame,prediction[0][0],(10, 120), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 3)
cv2.imshow("Frame", frame)
k = [prediction[0][0]]
interrupt = cv2.waitKey(10)
if interrupt & 0xFF == 27: # esc key
    break
cap.release()
cv2.destroyAllWindows()
print(k)
prediction_text =str(k)
speech_object = gTTS(text= prediction_text, lang= 'en', slow= False)
speech_object.save("myspeak.mp3")
os.system("start myspeak”)

