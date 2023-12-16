from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from keras.models import load_model
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, Flatten, Reshape, \
  Conv2D, Conv2DTranspose, UpSampling2D, BatchNormalization,  RandomRotation,  RandomFlip, \
  LeakyReLU, Dropout, Rescaling, Add, GlobalAveragePooling2D, MaxPooling2D, Concatenate, Activation, MaxPool2D, Convolution2D, LocallyConnected2D
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras import models
import random


def get_cnn_embeddings(image):
    #load the model
    model = load_model("./data/cnn_embed.h5")

    #get the embeddings for the input image

    #read the image
    X= np.array(image)
    print(X.shape)
    X.resize((47, 62))
    X = X.flatten()
    print(X.shape)
    X_reshaped = X.reshape((1, 62, 47))

    # embeddings
    intermediate_layer_name = 'conv2d_2'
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(intermediate_layer_name).output)
    embeddings = intermediate_layer_model.predict(X_reshaped)

    return embeddings

def display_matches(test_embeddings,image):
    data = np.load("./data/lfw-60-dataset.npz")
    train_x = data['arr_0']
    test_x = data['arr_2']
    embed_data = np.load("./data/lfw-embeddings_128.npz")
    train_embeddings = embed_data['arr_0']
    distances = cosine_similarity(test_embeddings.reshape(1,-1), train_embeddings.reshape(len(train_embeddings), -1))
    top5_indices = np.argsort(distances[0])[-5:][::-1]

    num_images = len(test_x)

    # Randomly choose three indices
    random_indices = random.sample(range(num_images), 3)

    plt.figure(figsize=(10, 5))

    for i, index in enumerate(random_indices):
        plt.subplot(1, 3, i + 1)
        plt.imshow(test_x[index])
        plt.title(f"Random Image {index}")
        plt.savefig(f'plt_outputs/curr_image_{i}.png')
        plt.clf()



    # Display the top 5 closest match images
    # for i, index in enumerate(top5_indices):
    #     # plt.subplot(2, 3, i + 2)
    #     plt.figure(figsize=(15,6))
    #     plt.imshow(train_x[index])
    #     plt.title(f'Top {i + 1}')
    #     plt.savefig(f'plt_outputs/curr_image_{i}.png')
    #     plt.clf()

