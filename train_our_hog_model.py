import numpy as np
from PIL import Image
from scipy import ndimage, datasets, spatial
import time
import math
from skimage import data, color, feature, transform
from sklearn.datasets import fetch_lfw_people
from sklearn.feature_extraction.image import PatchExtractor
from itertools import chain
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import cv2 as cv
import joblib


"""function returns gradient orientations in form of 9 bin histogram for the input window"""
def hog_patch(patch):
    x_grad = cv.Sobel(patch, cv.CV_64F, 1, 0, ksize=3)
    y_grad = cv.Sobel(patch, cv.CV_64F, 0, 1, ksize=3)
    mag, angle = cv.cartToPolar(x_grad, y_grad)
    hist, bins = np.histogram(angle, bins=9, range=(0, math.pi), weights=mag)
    return hist

"""Fuction return a HOG descriptor for input patch of an image"""
def hog_extractor(face_image):
    # face_image = cv.cvtColor(face_image, cv.COLOR_BGR2GRAY)
    # face_image = np.sqrt(face_image)
    (h, w) = face_image.shape
    h, w = h//8, w//8
    face_image = cv.resize(face_image, (w*8, h*8))

    hog_8by8 = []
    for i in range(h):
        r = i*8
        curr_r = []
        for j in range(w):
            c = j*8
            hist = hog_patch(face_image[r:r+8, c:c+8])
            curr_r.append(hist)
        hog_8by8.append(curr_r)

    hog_8by8 = np.array(hog_8by8)
    (h, w, c) = hog_8by8.shape

    hog_16by16 = []
    for i in range(h-1):
        curr_r = []
        for j in range(w-1):
            arr = hog_8by8[i:i+2, j:j+2].flatten()
            norm = np.linalg.norm(arr)
            arr = arr / (norm + 1e-6) if norm > 0 else arr
            curr_r.append(arr)
        hog_16by16.append(curr_r)
    hog_16by16 = np.array(hog_16by16)
    return hog_16by16.flatten()

#Obtain Positive Training Samples
faces = fetch_lfw_people()
positive_patches = faces.images
positive_patch_size = positive_patches[0].shape
print("Shape of positive patches ", positive_patches.shape)
#Obtain Negative Training Samples
imgs_to_use = ['camera', 'text', 'coins', 'moon',
            'page', 'clock', 'immunohistochemistry',
            'chelsea', 'coffee', 'hubble_deep_field']
images = [getattr(data, name)()
        for name in imgs_to_use]
for i in range(len(images)):
    if len(images[i].shape)==3:
        images[i] = color.rgb2gray(images[i])

def extract_patches(img, N, scale=1.0, patch_size=positive_patch_size):
    extracted_patch_size = tuple((scale * np.array(patch_size)).astype(int))
    extractor = PatchExtractor(patch_size=extracted_patch_size,
                            max_patches=N, random_state=0)
    patches = extractor.transform(img[np.newaxis])
    if scale != 1:
        patches = np.array([transform.resize(patch, patch_size)
                            for patch in patches])
    return patches

negative_patches = np.vstack([extract_patches(im, 1000, scale)
                        for im in images for scale in [0.5, 1.0, 2.0]])
print("Shape of negative patches ", negative_patches.shape)


#Combine Positive and Negative Samples and compute HOG
X_train = np.array([hog_extractor(im)
                    for im in chain(positive_patches,
                                    negative_patches)])
y_train = np.zeros(X_train.shape[0])
y_train[:positive_patches.shape[0]] = 1
print("positive negative merged ", X_train.shape)

# Impute missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)

# Training a Support Vector Machine
cross_val_score(GaussianNB(), X_train, y_train)
print("Cross validation score ", cross_val_score(GaussianNB(), X_train, y_train))

grid = GridSearchCV(LinearSVC(), {'C': [1.0, 2.0, 4.0, 8.0]})
grid.fit(X_train, y_train)
print("Best score ", grid.best_score_)
print("Best param ", grid.best_params_)

# Get the best hyperparameters
best_params = grid.best_params_
print("Best Hyperparameters:", best_params)
    
#save your model or results
joblib.dump(grid, 'our_hog_model.pkl', compress=True)

#load your model for further usage
joblib.load("our_hog_model.pkl")
    


























