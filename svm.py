
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from skimage import feature
from imutils import paths
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import cv2
import os
import pickle


basepath= os.path.normpath('C:\Users\wadhe\OneDrive\Desktop\updated_MediPharma breast cancer web 100%\updated_MediPharma breast cancer web 100%\updated_MediPharma breast cancer web 100%\MediPharma breast cancer web 100%\MediPharma')

def fd_hu_moments(image):
    #For Shape of signature Image
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

def quantify_image(image):
    #For Speed and pressure of signature image

    # compute the histogram of oriented gradients feature vector for
    # the input image
    features = feature.hog(image, orientations=9,
        pixels_per_cell=(10, 10), cells_per_block=(2, 2),
        transform_sqrt=True, block_norm="L1")

    # return the feature vector
    return features

def load_split(path):
    # grab the list of images in the input directory, then initialize
    # the list of data (i.e., images) and class labels
    path=trainingPath
    imagePaths = list(paths.list_images(path))
    data = []
    labels = []

    # loop over the image paths
    for imagePath in imagePaths:
        # extract the class label from the filename
        label = imagePath.split(os.path.sep)[-2]
#        print(imagePath)
        # load the input image, convert it to grayscale, and resize
        # it to 200x200 pixels, ignoring aspect ratio
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (64, 64))

        # threshold the image such that the drawing appears as white
        # on a black background
        image = cv2.threshold(image, 0, 255,
            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        # quantify the image
        features1 = quantify_image(image)
        features2 = fd_hu_moments(image)
        global_feature = np.hstack([features1,features2])

        # update the data and labels lists, respectively
        data.append(global_feature)

        labels.append(label)

    # return the data and labels
    return (np.array(data), np.array(labels))


trainingPath = os.path.sep.join([basepath, "training_set"])
testingPath = os.path.sep.join([basepath, "testing_set"])


## loading the training and testing data
#print("[INFO] loading data...")
(trainX, trainY) = load_split(trainingPath)
(testX, testY) = load_split(testingPath)

# encode the labels as integers
le = LabelEncoder()
trainY = le.fit_transform(trainY)
testY = le.transform(testY)

# initialize our trials dictionary
trials = {}

def SVM_Cl():

    C = 1.0
    model = svm.SVC(kernel='linear',C=C)

    # loop over the number of trials to run
    for i in range(0, 5):

        # train the model
        print("[INFO] training model {} of {}...".format(i + 1,
            5))  #args["trials"]))
        model = svm.SVC(kernel='linear',C=C)

        model.fit(trainX, trainY)

        # make predictions on the testing data and initialize a dictionary
        # to store our computed metrics
        predictions = model.predict(testX)
        metrics = {}
        # compute the confusion matrix and and use it to derive the raw
        # accuracy, sensitivity, and specificity
        #cm = confusion_matrix(testY, predictions).flatten()
        #(tn, fp, fn, tp) = cm
        # metrics["acc"] = (tp + tn) / float(cm.sum())
        # metrics["sensitivity"] = tp / float(tp + fn)
        # metrics["specificity"] = tn / float(tn + fp)

        # loop over the metrics
        # for (k, v) in metrics.items():
        #     # update the trials dictionary with the list of values for
        #     # the current metric
        #     l = trials.get(k, [])
        #     l.append(v)
        #     trials[k] = l



    with open(basepath + '/breast_cancer_svm.pkl', 'wb') as f:
            pickle.dump(model, f)

    A="SVM Accuracy: {0:.2%}".format(accuracy_score(predictions, testY))
    C="SVM Model Saved as <<  oral_cancer_svm.pkl  >>"

    D = A+'\n'+ C
    print(D)
    return D
SVM_Cl()