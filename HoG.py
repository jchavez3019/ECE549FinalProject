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

class HOG_face_detector:
    def __init__(self, input_image_path):
        self.test_image = cv.imread(input_image_path, 0)
        self.test_image = transform.rescale(self.test_image, 0.5)
        self.scale_range = [1.0, 2.0, 3.0]
        self.positive_patches = None
        self.negative_patches = None
        self.positive_patch_size = None
        self.grid = None

    """function returns gradient orientations in form of 9 bin histogram for the input window"""
    def hog_patch(self, patch):
        x_grad = cv.Sobel(patch, cv.CV_64F, 1, 0, ksize=3)
        y_grad = cv.Sobel(patch, cv.CV_64F, 0, 1, ksize=3)
        mag, angle = cv.cartToPolar(x_grad, y_grad)
        hist, bins = np.histogram(angle, bins=9, range=(0, math.pi), weights=mag)
        return hist

    """Fuction return a HOG descriptor for input patch of an image"""
    def hog_extractor(self, face_image):
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
                hist = self.hog_patch(face_image[r:r+8, c:c+8])
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

    """Function retrieves training data from sklearn and skimage"""
    def get_training_data(self):
        #Obtain Positive Training Samples
        faces = fetch_lfw_people()
        self.positive_patches = faces.images
        self.positive_patch_size = self.positive_patches[0].shape
        print("Shape of positive patches ", self.positive_patches.shape)
        #Obtain Negative Training Samples
        imgs_to_use = ['camera', 'text', 'coins', 'moon',
                    'page', 'clock', 'immunohistochemistry',
                    'chelsea', 'coffee', 'hubble_deep_field']
        images = [getattr(data, name)()
                for name in imgs_to_use]
        for i in range(len(images)):
            if len(images[i].shape)==3:
                images[i] = color.rgb2gray(images[i])

        def extract_patches(img, N, scale=1.0, patch_size=self.positive_patch_size):
            extracted_patch_size = tuple((scale * np.array(patch_size)).astype(int))
            extractor = PatchExtractor(patch_size=extracted_patch_size,
                                    max_patches=N, random_state=0)
            patches = extractor.transform(img[np.newaxis])
            if scale != 1:
                patches = np.array([transform.resize(patch, patch_size)
                                    for patch in patches])
            return patches
        
        self.negative_patches = np.vstack([extract_patches(im, 1000, scale)
                              for im in images for scale in [0.5, 1.0, 2.0]])
        print("Shape of negative patches ", self.negative_patches.shape)


    """Function trains a linear SVM using the training data"""
    def train_model(self):
        #Combine Positive and Negative Samples and compute HOG
        X_train = np.array([self.hog_extractor(im)
                            for im in chain(self.positive_patches,
                                            self.negative_patches)])
        y_train = np.zeros(X_train.shape[0])
        y_train[:self.positive_patches.shape[0]] = 1
        print("positive negative merged ", X_train.shape)

        # Impute missing values using SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        X_train = imputer.fit_transform(X_train)

        # Training a Support Vector Machine
        cross_val_score(GaussianNB(), X_train, y_train)
        print("Cross validation score ", cross_val_score(GaussianNB(), X_train, y_train))

        self.grid = GridSearchCV(LinearSVC(), {'C': [1.0, 2.0, 4.0, 8.0]})
        self.grid.fit(X_train, y_train)
        print("Best score ", self.grid.best_score_)
        print("Best param ", self.grid.best_params_)

        # Get the best hyperparameters
        best_params = self.grid.best_params_
        print("Best Hyperparameters:", best_params)
    
    """Function returns indices for all the patches of possible faces in an image"""
    def sliding_window(self, img, patch_size=None, istep=2, jstep=2, scale=1.0):
        if patch_size is None:
            patch_size = self.positive_patch_size
        Ni, Nj = (int(scale * s) for s in patch_size)
        indices, patches = [], []
        for i in range(0, img.shape[0] - Ni, istep):
            for j in range(0, img.shape[1] - Nj, jstep):
                patch = img[i:i + Ni, j:j + Nj]
                if scale != 1:
                    patch = transform.resize(patch, patch_size)
                indices.append((i, j))
                patches.append(patch)
        return indices, patches
    
    """Function uses the sliding window technique and the trained model to 
    distinguish faces from non-faces in an input image. It then plots a 
    bounding box around the face"""
    def detect_face_plot(self):
        curr_score = np.NINF
        curr_scale = 1.0
        winner_idx = None
        winner_indices = None
        for s in self.scale_range:
            indices, patches = self.sliding_window(self.test_image, patch_size=(self.positive_patch_size),
                            istep=2, jstep=2, scale=s)
            patches_hog = np.array([self.hog_extractor(patch) for patch in patches])
            # Get decision scores for each sample
            decision_scores = self.grid.decision_function(patches_hog)
            win_idx = np.argmax(decision_scores)
            if decision_scores[win_idx] > curr_score:
                curr_score = decision_scores[win_idx]
                curr_scale = s
                winner_idx = win_idx
                winner_indices = indices

        print("best score is ", curr_score)
        print("best scale ", curr_scale)
        print("best idx ", winner_idx)

        ########## Plot the bounding box ##################
        fig, ax = plt.subplots()
        ax.imshow(self.test_image, cmap='gray')
        ax.axis('off')
        (Ni, Nj) = tuple(x*int(curr_scale) for x in self.positive_patch_size)
        print(Ni, Nj)
        winner_indices = np.array(winner_indices)
        i, j = winner_indices[winner_idx]
        new_image = self.test_image[i:i+Ni, j:j+Nj]
        ax.add_patch(plt.Rectangle((j, i), Nj, Ni, edgecolor='red',
                                    alpha=0.5, lw=3, facecolor='none'))
        plt.show()

def main():
    input_image_path = "./face.jpg"
    hog_detector = HOG_face_detector(input_image_path)
    hog_detector.get_training_data()
    hog_detector.train_model()
    hog_detector.detect_face_plot()

if __name__ == "__main__":
    main()
    


























