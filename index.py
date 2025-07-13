import os
import cv2
import numpy as np
import time
from my_model import VGGNet
import sqlite3
import pickle
import imutils
import glob


if os.path.exists("another_database"):

    my_path = "another_database/"

    files = glob.glob(my_path + '/**/*.jpg', recursive=True)

    print("--------------------------------------------------")
    print("         feature extraction starts")
    print("--------------------------------------------------")

    feats = []
    names = []

    model = VGGNet()  # upload your model here

    t1 = time.time()

    for i, img_path in enumerate(files):

        norm_feat = model.extract_feat(img_path)

        Dir, Sub_dir, name = img_path.split("\\")
        p = Sub_dir + "/" + name
        print(p)

        feats.append(norm_feat)
        names.append(p)

        print("extracting feature from image No. %d , %d images in total" % ((i + 1), len(files)))

    feats = np.array(feats)

    t2 = time.time()

    print("Time taken for complete feature extraction = ", t2 - t1)

    print("--------------------------------------------------")
    print("      writing feature extraction results ...")
    print("--------------------------------------------------")

    if not os.path.exists("unknown/cn.pickle"):

        datta = {"dataset_1": feats, "dataset_2": np.string_(names)}
        f = open("unknown/cn.pickle", "wb")
        f.write(pickle.dumps(datta))
        f.close()
        print("Pickle file written successfully")

    else:

        the_list = pickle.loads(open("unknown/cn.pickle", "rb").read())

        print('Existing file loaded successfully')

        # append to the list
        np.append(the_list["dataset_1"], feats)
        np.append(the_list["dataset_2"], np.string_(names))

        print('Data appended successfully')

        # write back to file with the same name
        fh = open('unknown/cn.pickle', 'wb')
        fh.write(pickle.dumps(the_list))
        fh.close()

        print('Existing file again created with appended data successfully')

else:

    print("First create the sub folders manually")
