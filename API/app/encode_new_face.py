import imutils
from imutils import paths
import face_recognition
import pickle
import cv2
import os, sys, inspect
import numpy as np


'''Encoding a new face'''
def new_face(img_path, name):
     #check if encoded pickle file exists, otherwise write to new file
     if os.path.exists('encodings.pickle'):
         data = pickle.loads(open('encodings.pickle', "rb").read())
         knownData= data['encodings']
     else:
         knownData={}
 
     #convert image 
     img = cv2.imread(img_path)
     rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
     name = str(name)  #Identification name

     boxes = face_recognition.face_locations(rgb, model='hog')
     encoding = face_recognition.face_encodings(rgb, boxes)
     knownData[name]=encoding     

     #write encoding to pickle file
     data = {"encodings":knownData}
     f=open('encodings.pickle', "wb")
     f.write(pickle.dumps(data))
     f.close()
     return


'''Loss function to compare difference between new image and reference image'''
def compute_loss2(y_truth, y_est):
    value = np.sum(np.power(y_truth-y_est,2))/len(y_est) #check this
    return value


'''Identifying someone from a given picture'''
def identify(img_path):
     if not os.path.exists('encodings.pickle'):
         return ("No faces stored in system")
     data = pickle.loads(open('encodings.pickle', "rb").read())
     knownData= data['encodings']
     img = cv2.imread(img_path)
     rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
     rgb = imutils.resize(img, width=750)
     r = img.shape[1] / float(rgb.shape[1])

     # detect the (x, y)-coordinates of the bounding boxes
     # corresponding to each face in the input frame, then compute
     # the facial embeddings for each face
     boxes = face_recognition.face_locations(rgb, model = 'hog')
     encodings = face_recognition.face_encodings(rgb, boxes)
     names = []
     scores = []

     # loop over the facial encodings
     for encoding in encodings:
	# attempt to match each face in the input image to our known
	# encodings
	

        minDist = 100
        for person in knownData:
            
            value = compute_loss2(encoding, knownData[person])
            if value < minDist:
                  minDist = value
                  identity = person
                  print(identity, minDist)

        if minDist > 0.17:
             identity = "Unknown"

	
	# update the list of names
        names.append(identity)
        if identity is 'Unknown':
            scores.append(minDist)
        else:
            scores.append(float(minDist))
      
     #currently only returns identity of one person per image
     return(names[0]) 

