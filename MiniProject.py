import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import math
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
import csv
from PIL import Image
from numpy import savetxt
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import model_selection
from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score



    
path1 = glob.glob("C:/MINI_PROJECT_DATASET/01_palm/*.png")
path2 = glob.glob("C:/MINI_PROJECT_DATASET/02_l/*.png")
path3 = glob.glob("C:/MINI_PROJECT_DATASET/03_fist/*.png")
path4 = glob.glob("C:/MINI_PROJECT_DATASET/04_fist_moved/*.png")
path5 = glob.glob("C:/MINI_PROJECT_DATASET/05_thumb/*.png")
path6 = glob.glob("C:/MINI_PROJECT_DATASET/06_index/*.png")
path7 = glob.glob("C:/MINI_PROJECT_DATASET/07_ok/*.png")
path8 = glob.glob("C:/MINI_PROJECT_DATASET/08_palm_moved/*.png")
path9 = glob.glob("C:/MINI_PROJECT_DATASET/09_c/*.png")
path10 = glob.glob("C:/MINI_PROJECT_DATASET/10_down/*.png")


pixels=[]
y_value = []

for file in path1:
    Im = Image.open(file)
    pixels.append(list(Im.getdata()))
    y_value.append(1)

for file in path2:
    Im = Image.open(file)
    pixels.append(list(Im.getdata()))
    y_value.append(2)


for file in path3:
    Im = Image.open(file)
    pixels.append(list(Im.getdata()))
    y_value.append(3)

for file in path4:
    Im = Image.open(file)
    pixels.append(list(Im.getdata()))
    y_value.append(4)


for file in path5:
    Im = Image.open(file)
    pixels.append(list(Im.getdata()))
    y_value.append(5)



for file in path6:
    Im = Image.open(file)
    pixels.append(list(Im.getdata()))
    y_value.append(6)



for file in path7:
    Im = Image.open(file)
    pixels.append(list(Im.getdata()))
    y_value.append(7)



for file in path8:
    Im = Image.open(file)
    pixels.append(list(Im.getdata()))
    y_value.append(8)



for file in path9:
    Im = Image.open(file)
    pixels.append(list(Im.getdata()))
    y_value.append(9)

for file in path10:
    Im = Image.open(file)
    pixels.append(list(Im.getdata()))
    y_value.append(10)


pixels_arr=np.asarray(pixels)
print(" The shape of the dataset is which is created from Images is -> ")
print(pixels_arr.shape)

    
x, y = shuffle(pixels, y_value)
x = preprocessing.normalize(x)


X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.7 )
X_test, X_validation, y_test, y_validation = train_test_split(X_test, y_test, test_size=0.14)

lm = linear_model.LogisticRegression(multi_class='ovr', solver='liblinear')
lm.fit(X_train, y_train)


predicted = lm.predict(X_test)
matrix = confusion_matrix(y_test, predicted)
print("The confusion matrix is -> ")
print(matrix)



report = classification_report(y_test, predicted)
print("The classification report for the Model is -> ")
print(report)


score =accuracy_score(y_test,predicted)


print("The Accuracy score for the model is  -> ")

print(score)





import cv2
import numpy as np
import math

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
     
while(cap.isOpened()):
        
    try:  #an error comes if it does not find anything in window as it cannot find contour of max area
          #therefore this try error statement
          
        ret, frame = cap.read()
        
        frame=cv2.flip(frame,1)
        kernel = np.ones((3,3),np.uint8)
        
        #define region of interest
        roi=frame[100:340, 0:740]
        
        
        cv2.rectangle(frame,(100,100),(740,340),(0,255,0),0)   
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

             
         
    # define skin color range in HSV
        min_HSV = np.array([0, 48, 80], dtype = "uint8")
        max_HSV = np.array([20, 255, 255], dtype = "uint8")
        #lower_skin = np.array([0,20,70], dtype=np.uint8)
        #upper_skin = np.array([20,255,255], dtype=np.uint8)
         
        
        mask = cv2.inRange(hsv, min_HSV, max_HSV)
        
   
        
    #extrapolate the hand to fill dark spots within
        mask = cv2.dilate(mask,kernel,iterations = 4)
        
    #blur the image
        mask = cv2.GaussianBlur(mask,(5,5),100) 
          
    
        cv2.imwrite(r'C:\Users\uditr\Documents\5th sem\ML\Frame1'+'.png', mask)


        pixels=[]
        Im = Image.open(r'C:\Users\uditr\Documents\5th sem\ML\Frame1.png')
        pixels.append(list(Im.getdata()))
   

        font = cv2.FONT_HERSHEY_SIMPLEX

        if lm.predict(pixels) == 1:
            cv2.putText(frame,'PALM',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)


        elif lm.predict(pixels) == 2:   
            cv2.putText(frame,'L',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)

        elif lm.predict(pixels) == 3:   
            cv2.putText(frame,'FIST',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA) 

        elif lm.predict(pixels) == 4:   
            cv2.putText(frame,'FIST MOVED',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA) 

        elif lm.predict(pixels) == 5:   
            cv2.putText(frame,'THUMB',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)    

        elif lm.predict(pixels) == 6:   
            cv2.putText(frame,'INDEX',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)           

        elif lm.predict(pixels) == 7:   
            cv2.putText(frame,'OK',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)    

        elif lm.predict(pixels) == 8:   
            cv2.putText(frame,'PALM MOVED',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)    

        elif lm.predict(pixels) == 9:   
            cv2.putText(frame,'C',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)    

        elif lm.predict(pixels) == 10:   
            cv2.putText(frame,'DOWN',(0,50), font, 2, (0,0,255), 3, cv2.LINE_AA)    


        cv2.imshow('mask',mask)
        cv2.imshow('frame',frame)

    except:
        pass

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


    
    

cv2.destroyAllWindows()
cap.release()

    






  
