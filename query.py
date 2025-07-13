# USAGE
# python query.py --query 01.jpg

import os
import cv2
from my_model import VGGNet
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
from collections import defaultdict
import sqlite3
import datetime
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import pickle
from PIL import Image


ap = argparse.ArgumentParser()
ap.add_argument("--query", required=True, help="Path to query which contains image to be queried")
args = vars(ap.parse_args())


def getProfile(id):

    conn = sqlite3.connect("FaceBase.db")

    cmd = "SELECT * FROM unknown_person where ID="+str(id)

    cursor = conn.execute(cmd)

    profile = None

    for row in cursor:

        profile = row

    conn.close()

    return profile


def insertdata(Id, Age, Gender, Location, Time, Expression, Age_group, Confidence, Count):

    conn = sqlite3.connect("FaceBase.db")

    cmd = "SELECT * FROM face_detection WHERE person_ID="+str(Id)

    cursor = conn.execute(cmd)

    isRecordExist = 0

    for row in cursor:

        isRecordExist = 1

    if isRecordExist == 1:

        cmd = "UPDATE face_detection SET Location="+str(Location)+" WHERE person_ID="+str(Id)

    else:

        conn.execute("INSERT INTO face_detection (person_ID, Age, Gender, Location, Time, Expression, Age_group, Confidence, Count) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (str(Id), str(Age), str(Gender), str(Location), str(Time), str(Expression), str(Age_group), str(Confidence), str(Count)))

        conn.commit()

        print("New record inserted")


def load_caffe_models():

    age_net = cv2.dnn.readNetFromCaffe('Age/deploy_age.prototxt', 'Age/age_net.caffemodel')

    return age_net


age_net = load_caffe_models()


# Loading age model
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']


# Loading our custom model for facial expression
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]


# read in indexed images' feature vectors and corresponding image names
the_list = pickle.loads(open("unknown/cn.pickle", "rb").read())
feats = the_list['dataset_1'][:]
imgNames = the_list['dataset_2'][:]
print("Total names in the trained model file: ",imgNames)


print("--------------------------------------------------")
print("               searching starts")
print("--------------------------------------------------")

# read and show query image
queryDir = args["query"]
queryImg = mpimg.imread(queryDir)

plt.title("Query Image")
plt.imshow(queryImg)
plt.show()

# init VGGNet16 model
model = VGGNet()

# extract query image's feature, compute simlarity score and sort
queryVec = model.extract_feat(queryDir)
scores = np.dot(queryVec, feats.T)
rank_ID = np.argsort(scores)[::-1]
rank_score = scores[rank_ID]

# number of top retrieved images to show
maxres = 1
imlist = [imgNames[index] for i, index in enumerate(rank_ID[0:maxres])]
print("Top %d image in order are: " % maxres, imlist)

# show top maxres retrieved result one by one
for i, im in enumerate(imlist):

    f = str(im, 'utf-8')

    m = "another_database" + "/" + f

    image = mpimg.imread(m)

    name, ext = f.split(".")

    print(name)

    subfolder, im_n = name.split("/")
    print(subfolder)

    profile = getProfile(subfolder)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(gray, (64, 64))
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)

    preds = emotion_classifier.predict(roi)[0]

    label = EMOTIONS[preds.argmax()]

    text = "{}".format(label)

    blob = cv2.dnn.blobFromImage(image, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

    # Predict Age
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = age_list[age_preds[0].argmax()]

    Time = datetime.datetime.now().strftime("%I:%M:%S%p")

    Id = profile[0]
    Name = profile[1]
    gender = profile[3]
    age_group = profile[5]
    confidence = "{:.2%}".format(rank_score[0])

    print("\n")

    print("Id:", Id)
    print("Name: ",Name)
    print("Age: ",age)
    print("Gender: ",gender)
    print("Age Group: ",age_group)
    print("Expression: ",text)
    print("Confidence: ",confidence)
    print("Time: ",Time)

    print("\n")

    insertdata(Id, age, gender, queryDir, Time, text, age_group, confidence, "_")
    
    plt.title("Most closest output %d" % (i + 1))
    plt.imshow(image)
    plt.show()
