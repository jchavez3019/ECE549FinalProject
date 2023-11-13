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
from sklearn.svm import LinearSVC
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier 

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import cv2 as cv





def preprocessing():
    pass
def gradient_images(img):
    im = np.float32(im)/255.0

    #gradients
    x_grad = cv.Sobel(img, )

    cv.HOGDescriptor()
    pass


"""function returns the dominant gradient orientation within the provided window"""
def hog_patch(patch):
    x_grad = cv.Sobel(patch, cv.CV_64F, 1, 0, ksize=3)
    y_grad = cv.Sobel(patch, cv.CV_64F, 0, 1, ksize=3)
    mag, angle = cv.cartToPolar(x_grad, y_grad)
    hist, bins = np.histogram(angle, bins=9, range=(0, math.pi), weights=mag)
    return hist


def hog_extractor(face_image):

    # face_image = cv.cvtColor(face_image, cv.COLOR_BGR2GRAY)
    # # (h, w) = img_gray.shape
    # # h, w = h//8, w//8

    # winSize = (64, 128)
    # blockSize = (16,16)
    # blockStride = (8,8)
    # cellSize = (8,8)
    # nbins = 9
    # derivAperture = 1
    # winSigma = 4.
    # histogramNormType = 0
    # L2HysThreshold = 2.0000000000000001e-01
    # gammaCorrection = 0
    # nlevels = 64
    # hog = cv.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
    #                         histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
    # #compute(img[, winStride[, padding[, locations]]]) -> descriptors
    # winStride = (8,8)
    # padding = (8,8)
    # locations = ((10,20),)
    # hist = hog.compute(face_image,winStride,padding,locations)
    # return hist

    # # print ('HOG Descriptor:', hist)
    # # print ('HOG Descriptor has shape:', hist.shape)



    # face_image = cv.cvtColor(face_image, cv.COLOR_BGR2GRAY)
    # (h, w) = face_image.shape
    # h, w = h//8, w//8
    # face_image = cv.resize(face_image, (w*8, h*8))
    # h, w = 128//8, 64//8
    face_image = np.sqrt(face_image)
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
            arr = arr/norm
            curr_r.append(arr)
        hog_16by16.append(curr_r)
    hog_16by16 = np.array(hog_16by16)

    return hog_16by16.flatten()


# im = cv.imread("face.jpg")
# hog_computed = hog_extractor(im)
# print(hog_computed.shape)


#Training

#Obtain Positive Training Samples
faces = fetch_lfw_people()
positive_patches = faces.images
# positive_patches = positive_patches[:5000]
print(positive_patches.shape)
# print(positive_patches[0].shape)
# plt.imshow(positive_patches[0])
# plt.show()

#Obtain Negative Training Samples
imgs_to_use = ['camera', 'text', 'coins', 'moon',
               'page', 'clock', 'immunohistochemistry',
               'chelsea', 'coffee', 'hubble_deep_field']
images = [getattr(data, name)()
          for name in imgs_to_use]
for i in range(len(images)):
    if len(images[i].shape)==3:
        images[i] = color.rgb2gray(images[i])

def extract_patches(img, N, scale=1.0, patch_size=positive_patches[0].shape):
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
# negative_patches = negative_patches[:10000]
print(negative_patches.shape)

# fig, ax = plt.subplots(6, 10)
# for i, axi in enumerate(ax.flat):
#     axi.imshow(negative_patches[500 * i], cmap='gray')
#     axi.axis('off')
# plt.show()

#Combine Positive and Negative Samples and compute HOG
X_train = np.array([hog_extractor(im)
                    for im in chain(positive_patches,
                                    negative_patches)])
y_train = np.zeros(X_train.shape[0])
y_train[:positive_patches.shape[0]] = 1
print(X_train.shape)


#Training a Support Vector Machine
# cross_val_score(GaussianNB(), X_train, y_train)
# print(cross_val_score(GaussianNB(), X_train, y_train))

# grid = GridSearchCV(LinearSVC(), {'C': [1.0, 2.0, 4.0, 8.0]})
grid = HistGradientBoostingClassifier().fit(X_train, y_train)

print(grid.score)

our_model = grid
our_model.fit(X_train, y_train)



def sliding_window(img, patch_size=positive_patches[0].shape,
                   istep=2, jstep=2, scale=1.0):
    Ni, Nj = (int(scale * s) for s in patch_size)
    for i in range(0, img.shape[0] - Ni, istep):
        for j in range(0, img.shape[1] - Ni, jstep):
            patch = img[i:i + Ni, j:j + Nj]
            if scale != 1:
                patch = transform.resize(patch, patch_size)
            yield (i, j), patch


test_image = cv.imread("two_faces.webp", 0)
test_image = transform.rescale(test_image, 0.5)
# test_image = test_image[:250, 100:400]
# plt.imshow(test_image)
# plt.show()


# test_image = data.astronaut()
# test_image = color.rgb2gray(test_image)
# test_image = transform.rescale(test_image, 0.5)
# test_image = test_image[:160, 40:180]

indices, patches = zip(*sliding_window(test_image))
patches_hog = np.array([hog_extractor(patch) for patch in patches])
print(patches_hog.shape)


labels = our_model.predict(patches_hog)
print(labels.sum())



fig, ax = plt.subplots()
ax.imshow(test_image, cmap='gray')
ax.axis('off')

Ni, Nj = positive_patches[0].shape
indices = np.array(indices)

for i, j in indices[labels == 1]:
    ax.add_patch(plt.Rectangle((j, i), Nj, Ni, edgecolor='red',
                               alpha=0.3, lw=2, facecolor='none'))
    
plt.show()









