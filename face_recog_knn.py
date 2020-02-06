import math
from sklearn import neighbors
import os
import os.path
import pickle
import face_recognition
from imutils import paths
import cv2



def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    imagePaths = []

    for r,d,f in os.walk(train_dir):
        for file in f:
            imagePaths.append(str(r)+'/'+str(file))
    
    x=[]
    y=[]

    for (i, imagePath) in enumerate(imagePaths):
        print("[INFO] processing image {}/{}".format(i + 1,
		len(imagePaths)))
        name = imagePath.split(os.path.sep)[-2]
        
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes = face_recognition.face_locations(rgb, model='hog')
        encodings = face_recognition.face_encodings(rgb, boxes)
  
        for encoding in encodings:
            x.append(encoding)
            y.append(name)
    print(len(x), len(y))

    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(x))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(x, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf

def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.4):
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)
    
    X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]



if __name__=="__main__":
    #classifier = train("dataset2", model_save_path="trained_knn_model.clf", n_neighbors=2)

    imageFiles = []
    test_dir = "test"
    for r,d,f in os.walk(test_dir):
        for file in f:
            imageFiles.append(str(r)+'/'+str(file))
    
    for image in imageFiles:
        name_og = image.split(os.path.sep)[-2]
        predictions = predict(image, model_path="trained_knn_model.clf")
        for name, (top, right, bottom, left) in predictions:
            print("- Found {} at ({}, {})".format(name, left, top))
            print(name_og, '/n')


    
