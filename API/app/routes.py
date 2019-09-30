from app import app
from flask import Flask, redirect, url_for, request, render_template, send_from_directory
import json
import time
from datetime import datetime
import subprocess
import random
from encode_new_face import new_face, identify
import os, logging
from werkzeug import secure_filename


#Main Page
@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html")

@app.route("/findUser")
def findUser():
    return render_template("recognition.html")




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
