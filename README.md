# Facial Identification System
This repository contains code to implement a simple identification software that uses facial recognition and machine learning to identify a person. 

## Requirements
* [dlib](http://dlib.net/)
* [face_recognition](https://github.com/ageitgey/face_recognition)
* [OpenCV](https://www.pyimagesearch.com/opencv-tutorials-resources-guides/)
* [imutils](https://github.com/jrosebr1/imutils)

## Storing New Faces
Add pictures of people you want to be able to identify into the Database folder. Make sure that there is only one visible face in each image that you store. Label the folder with the person's name. You only need to add one image per person. 

Once you have added the images to the dataset folder, run the following script. 
```python encodeFaces.py --dataset dataset --encodings encodings.pickle --detection-method hog
```
This script will create encodings of each face in your dataset and store them as a Pickle file. 

## Running the Application with a WebCam
You can run the application and get it to recognize faces in real-time through your webcam. 

```python faces_video_ver2.py --encodings encodings.pickle --output webcam.avi --detection-method hog
```
## Acknowledgements



