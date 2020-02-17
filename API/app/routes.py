from app import app
from flask import Flask, redirect, url_for, request, render_template, send_from_directory
import json
import time
from datetime import datetime
import subprocess
import random
#from encode_new_face import new_face, identify
import os, logging
from werkzeug.utils import secure_filename

import imutils
from imutils import paths
import face_recognition
import pickle
import cv2
import os, sys, inspect
import numpy as np


#Main Page
@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html")

@app.route("/findUser")
def findUser():
    return render_template("recognition.html")


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



def create_new_folder(local_dir):
    newpath = local_dir
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    return newpath


'''API to add a new face to the system'''
@app.route('/pic', methods = ['POST'])
def api_root():
    if request.method == 'POST' and request.files['image']:
        img = request.files['image']
        img_name = secure_filename(img.filename)
        create_new_folder(app.config['UPLOAD_FOLDER'])
        saved_path = os.path.join(app.config['UPLOAD_FOLDER'], img_name)
        img.save(saved_path)
        name = request.form.get('person')
        new_face(saved_path, name)
        return send_from_directory(app.config['UPLOAD_FOLDER'],img_name, as_attachment=True)
    else:
        return "Where is the image?"


'''API to identify a person from a given picture'''
@app.route('/recognize', methods = ['POST'])
def recognize():
    if request.method == 'POST' and request.files['image']:
        img = request.files['image']
        img_name = secure_filename(img.filename)
        create_new_folder(app.config['UPLOAD_FOLDER'])
        saved_path = os.path.join(app.config['UPLOAD_FOLDER'], img_name)
        img.save(saved_path)
        name = identify(saved_path)
        return name
    else:
        return "Where is the image?"
