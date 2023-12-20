from flask import Flask, request, render_template, jsonify
from flask_socketio import SocketIO
from flask_cors import CORS
import cv2
from PIL import Image
import os
from imageProcessing import returnDetectedFaces
import shutil
import base64
from io import BytesIO
from get_cnn_matches import get_cnn_embeddings, display_matches

# from facenet_pytorch import MTCNN
import pickle
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm
import os
from os import listdir
from numpy import load
from numpy import asarray
from numpy import savez_compressed
from sklearn.preprocessing import StandardScaler as StandardScaler_C
from sklearn.preprocessing import MinMaxScaler as MinMaxScaler_C
from sklearn.neighbors import KNeighborsClassifier as KNeighborsClassifier_C
from sklearn.preprocessing import LabelEncoder as LabelEncoder_C
from sklearn.pipeline import Pipeline
from sklearn import metrics as metrics_C
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.image import PatchExtractor
from itertools import chain
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split

from sklearn.svm import LinearSVC
from sklearn.decomposition import IncrementalPCA
import joblib
import pandas as pd
import re

class LFW_IncrementalPCA(IncrementalPCA):
    def __init__(self, dataset_paths: list[str], n_components: int, batch_size: int = 16, whiten: bool = True, encode_labels=True):
        super(LFW_IncrementalPCA, self).__init__(
            n_components=n_components,
            batch_size = batch_size,
            whiten=whiten
        )
        # pca = IncrementalPCA(n_components=n_components,batch_size = batch_size)
        self.dataset_paths = dataset_paths
        self.le = None
        self.encode_labels = encode_labels
        if encode_labels:
          labels = []
          for path in dataset_paths:
            data = load(path)
            print(data['arr_1'].shape)
            print(data['arr_3'].shape)
            labels.extend(data['arr_1'].ravel())
            labels.extend(data['arr_3'].ravel())

          self.le = LabelEncoder_C()
          self.le.fit(labels)
          print(f'LabelEncoder Classes: {self.le.classes_}')


    def start_fit(self, display_progress: bool=False):
        r""" Begins an incremental fit on the given dataset paths
            Args:
                display_progress (bool): Whether to display incremental progress of the fit

            Returns:
                None
        """

        for i, file in enumerate(self.dataset_paths):
            data = load(file)
            X_train = data['arr_0']
            if display_progress:
                print(f"({i}, {file} | X_train.shape {X_train.shape})")
            X_train = X_train.reshape(X_train.shape[0], -1)
            # mean = X_train.mean()
            # std = X_train.std()
            # X_train = (X_train - mean) / std
            if (display_progress):
                print(f"i = {i} | Partial fit on X_train of sizes {X_train.shape}")
            self.partial_fit(X_train)

    def get_embeddings(self, display_progress: bool=False):
        r""" Transforms the given datasets into embeddings and returns them
            Args:
                display_progress (bool): Whether to display final progress of embeddings

            Returns:
                X_train_pca (np.array): X_train pca embeddings
                Y_train_pca (np.array): Y_train labels
                X_test_pca (np.array): X_test pca embeddings
                Y_test_pca (np.array): Y_test labels
        """

        X_train_pca = np.zeros((0, self.n_components_))
        X_test_pca = np.zeros((0, self.n_components_))
        Y_train_pca = np.zeros((0, 1))
        Y_test_pca = np.zeros((0, 1))

        for i, file in enumerate(self.dataset_paths):
            data = load(file)
            X_train, Y_train, X_test, Y_test = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
            if self.le is not None:
              Y_train = self.le.transform(Y_train)
              Y_test = self.le.transform(Y_test)
            # train_mean = X_train.mean()
            # train_std = X_train.std()
            # X_train = (X_train - train_mean) / train_std
            # test_mean = X_test.mean()
            # test_std = X_test.std()
            # X_test = (X_test - test_mean) / test_std
            if display_progress:
                print(f"({i}, {file} | X_train.shape {X_train.shape}, Y_train.shape {Y_train.shape}, X_test.shape {X_test.shape}, Y_test.shape {Y_test.shape})")
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_test = X_test.reshape(X_test.shape[0], -1)
            Y_train = Y_train.reshape(-1,1)
            Y_test = Y_test.reshape(-1,1)

            X_train_pca = np.vstack((X_train_pca, self.transform(X_train)))
            X_test_pca = np.vstack((X_test_pca, self.transform(X_test)))

            Y_train_pca = np.vstack((Y_train_pca, Y_train))
            Y_test_pca = np.vstack((Y_test_pca, Y_test))

        Y_train_pca = Y_train_pca.ravel()
        Y_test_pca = Y_test_pca.ravel()

        if display_progress:
            print(f"X_train_pca shape {X_train_pca.shape} | X_test_pca {X_test_pca.shape} | Y_train_pca shape {Y_train_pca.shape} | Y_test_pca {Y_test_pca.shape}")
        return np.array(X_train_pca), np.array(Y_train_pca), np.array(X_test_pca), np.array(Y_test_pca)

    def save_embeddings(self, save_path: str, display_progress: bool=False):
        r""" Transforms the given datasets into embeddings and returns them
            Args:
            save_path (str): npz file path to save the embeddings to
                display_progress (bool): Whether to display final progress of embeddings

            Returns:
                None
        """

        X_train_pca = np.zeros((0, self.n_components_))
        X_test_pca = np.zeros((0, self.n_components_))
        Y_train_pca = np.zeros((0, 1))
        Y_test_pca = np.zeros((0, 1))

        for i, file in enumerate(self.dataset_paths):
            data = load(file)
            X_train, Y_train, X_test, Y_test = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
            if self.le is not None:
              Y_train = self.le.transform(Y_train)
              Y_test = self.le.transform(Y_test)
            # train_mean = X_train.mean()
            # train_std = X_train.std()
            # X_train = (X_train - train_mean) / train_std
            # test_mean = X_test.mean()
            # test_std = X_test.std()
            # X_test = (X_test - test_mean) / test_std
            if display_progress:
                print(f"({i}, {file} | X_train.shape {X_train.shape}, Y_train.shape {Y_train.shape}, X_test.shape {X_test.shape}, Y_test.shape {Y_test.shape})")
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_test = X_test.reshape(X_test.shape[0], -1)
            Y_train = Y_train.reshape(-1,1)
            Y_test = Y_test.reshape(-1,1)

            X_train_pca = np.vstack((X_train_pca, self.transform(X_train)))
            X_test_pca = np.vstack((X_test_pca, self.transform(X_test)))

            Y_train_pca = np.vstack((Y_train_pca, Y_train))
            Y_test_pca = np.vstack((Y_test_pca, Y_test))

        Y_train_pca = Y_train_pca.ravel()
        Y_test_pca = Y_test_pca.ravel()

        if display_progress:
            print(f"X_train_pca shape {X_train_pca.shape} | X_test_pca {X_test_pca.shape} | Y_train_pca shape {Y_train_pca.shape} | Y_test_pca {Y_test_pca.shape}")
        savez_compressed(save_path, X_train_pca, Y_train_pca, X_test_pca, Y_test_pca)

    def display_embedding_space(self):
        X_train_pca = np.zeros((0, self.n_components_))

        for i, file in enumerate(self.dataset_paths):
            data = load(file)
            X_train = data['arr_0']
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_train_pca = np.vstack((X_train_pca, self.transform(X_train)))

        df2d = pd.DataFrame(X_train_pca, columns=list('xy'))

        # Plot Data Visualization (Matplotlib)
        df2d.plot(kind='scatter', x='x', y='y')
        plt.show()

    def show_eigen_faces(self):
      eigen_faces = self.components_
      # Show the first 16 eigenfaces
      fig, axes = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(8, 10))
      for i in range(16):
          axes[i % 4][i // 4].imshow(eigen_faces[i].reshape((data.shape[1], data.shape[2])), cmap="gray")
      plt.suptitle("Eigen Faces", fontsize=16)
      plt.show()

    def plot_projection_n_target(self, n = 8):
      for i, file in enumerate(self.dataset_paths):
            data = load(file)
            X_train = data['arr_0']
            y_train = data['arr_1']
            X_train = X_train.reshape(X_train.shape[0], -1)
      X_pca = self.transform(X_train)
      target = self.le.transform(y_train)
      number_of_people=n
      index_range=number_of_people*10
      fig=plt.figure(figsize=(10,8))
      ax=fig.add_subplot(1,1,1)
      scatter=ax.scatter(X_pca[:index_range,0],
                  X_pca[:index_range,1],
                  c=target[:index_range],
                  s=10,
                cmap=plt.get_cmap('jet', number_of_people)
                )
      ax.set_xlabel("First Principle Component")
      ax.set_ylabel("Second Principle Component")
      ax.set_title("PCA projection of {} people".format(number_of_people))
      fig.colorbar(scatter)

    def predict_result(self, testimages_path, our_model):
        print("Predicting on ", testimages_path)
        x_input = cv2.imread(testimages_path, 0)
        x_input_pca = self.transform(x_input.reshape(1,-1))
        output = our_model.predict(x_input_pca)
        label = self.le.inverse_transform(output)
        return label




trainingImagePath = "./uploaded_images/trainingDataSet/"
testingImagePath = "./uploaded_images/testingDataSet/"
LFWSampleImagePath = '/home/jorgejc2/Documents/ClassRepos/CS549FinalProjectFrontEnd/mtcnn_extracted_faces/Akbar_Hashemi_Rafsanjani/'
"/home/jorgejc2/Documents/ClassRepos/CS549FinalProjectFrontEnd/CS549Backend/uploaded_images/trainingDataSet"

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="http://localhost:4200")
CORS(app, 
    resources={r"/upload/*": {"origins": "http://localhost:4200"}, 
                     r"/get-training-images/*": {"origins": "http://localhost:4200"},
                     r"/get-testing-images/*": {"origins": "http://localhost:4200"},
                     r"/get-training-image-faces/*": {"origins": "http://localhost:4200"},
                     r"/get-testing-image-faces/*": {"origins": "http://localhost:4200"},
                     r"/get-sample-image-faces/*": {"origins": "http://localhost:4200"},
                     r"/flask_sockets/*": {"origins": "http://localhost:4200"},
                     r"/get-predicted-label/*": {"origins": "http://localhost:4200"},
                     r"/socket.io/*": {"origins": "http://localhost:4200"}
                     })

"""
    Routes used by the Dashboard/Main page
"""
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':

        if 'isTrainingData' not in request.form:
            response = jsonify({'error': 'No isTrainingData part'})
            response.status_code = 400  # Set an appropriate error status code
            return response
        
        if 'images' not in request.files:
            response = jsonify({'error': 'No file part'})
            response.status_code = 400  # Set an appropriate error status code
            return response
    
        # file = request.files['file']
        uploaded_images = request.files.getlist('images')
        isTrainingDataStr = request.form.get('isTrainingData')
        print("isTrainingDataStr is {}".format(isTrainingDataStr))
        if (isTrainingDataStr == 'true'):
            isTrainingData = True
            toSavePath = 'uploaded_images/' + 'trainingDataSet/'
        else: 
            isTrainingData = False
            toSavePath = 'uploaded_images/' + 'testingDataSet/'

        # delete previous contents if folder if exists 
        shutil.rmtree(toSavePath, ignore_errors=True)
            
        print("Received list of images in Flask; Uploading files to {} folder".format(toSavePath))

        # create a directory to hold the files if it does not exist already 
        if not os.path.exists('uploaded_images/'):
                os.makedirs('uploaded_images/')

        if not os.path.exists('uploaded_images/testingDataSet/'):
                os.makedirs('uploaded_images/testingDataSet/')

        if not os.path.exists('uploaded_images/trainingDataSet/'):
                os.makedirs('uploaded_images/trainingDataSet/')

        for img in uploaded_images:
            if img and allowed_file(img.filename):      
                img.save(os.path.join(toSavePath, img.filename))
                image = Image.open(os.path.join(toSavePath, img.filename))   
                curr_img_embedding = get_cnn_embeddings(image)
                display_matches(curr_img_embedding, image)

        

        # Process the uploaded file
        # You can save it, manipulate it, etc.
        # For this example, we'll return a success message
        return jsonify({'message': 'File uploaded successfully'})
    
    return render_template('upload.html')

@app.route('/get-training-images', methods=['GET'])
def get_training_images():
    start_index = request.args.get('startIndex', type=int, default=0)
    end_index = start_index + 20
    image_filenames = os.listdir(trainingImagePath)
    image_filenames = image_filenames[start_index:end_index]
    images = []
    for image_path in image_filenames:
        with open(trainingImagePath + image_path, 'rb') as img_file:
            encoded_img = base64.b64encode(img_file.read()).decode('utf-8')
            images.append(encoded_img)

    return jsonify(images)
     
@app.route('/get-testing-images', methods=['GET'])
def get_testing_images():
    start_index = request.args.get('startIndex', type=int, default=0)
    end_index = start_index + 20
    image_filenames = os.listdir(testingImagePath)
    image_filenames = image_filenames[start_index:end_index]
    images = []
    for image_path in image_filenames:
        with open(testingImagePath + image_path, 'rb') as img_file:
            encoded_img = base64.b64encode(img_file.read()).decode('utf-8')
            images.append(encoded_img)

    return jsonify(images)

@app.route('/get-training-image-faces', methods=['GET'])
def get_training_image_faces():
    start_index = request.args.get('startIndex', type=int, default=0)
    image_filenames = os.listdir(trainingImagePath)
    image_filename = image_filenames[start_index]


    face_images = returnDetectedFaces(trainingImagePath + image_filename)
    final_images = []

    for image in face_images:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        encoded_img = base64.b64encode(buffered.getvalue()).decode('utf-8')
        final_images.append(encoded_img)

    return jsonify(final_images)
     
@app.route('/get-testing-image-faces', methods=['GET'])
def get_testing_image_faces():
    start_index = request.args.get('startIndex', type=int, default=0)
    image_filenames = os.listdir(testingImagePath)
    image_filename = image_filenames[start_index]


    face_images = returnDetectedFaces(testingImagePath + image_filename)
    final_images = []

    for image in face_images:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        encoded_img = base64.b64encode(buffered.getvalue()).decode('utf-8')
        final_images.append(encoded_img)

    return jsonify(final_images)

"""
    Routes used by the LFW page
"""

@app.route('/get-sample-image-faces', methods=['GET'])
def get_sample_images():
    start_index = request.args.get('startIndex', type=int, default=0)
    end_index = start_index + 3
    image_filenames = os.listdir(LFWSampleImagePath)
    image_filenames = image_filenames[start_index:end_index]
    images = []
    for image_path in image_filenames:
        # Image.open(LFWSampleImagePath + image_path).show()
        with open(LFWSampleImagePath + image_path, 'rb') as img_file:
            encoded_img = base64.b64encode(img_file.read()).decode('utf-8')
            images.append(encoded_img)

    ret_dict = {
        'images': images,
        'image_paths': image_filenames
    }

    return jsonify(ret_dict)

@app.route('/get-predicted-label', methods=['GET'])
def get_sample_image_prediction():
    file_path = LFWSampleImagePath + request.args.get('startIndex', type=str, default="")
    print("Opening at file path ", file_path)
    # curr_img_embedding = get_cnn_embeddings(image)
    # display_matches(curr_img_embedding, image)
    inc_pca = LFW_IncrementalPCA([''], 10)
    with open('/home/jorgejc2/Documents/ClassRepos/CS549FinalProjectFrontEnd/CS549Backend/data/our_linearsvc_pca_model.pkl', 'r') as file:
        label = inc_pca.predict_result(file_path, pickle.load(file))

    return jsonify(label)


def allowed_file(filename):
    # Implement a function to check the allowed file extensions
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png', 'gif'}

# functions for debugging
@socketio.on('delayed_request')
def handle_delayed_request():
    # Simulate a delay of 5 seconds asynchronously
    print("Entered delayed response function")
    socketio.sleep(5)
    socketio.emit('delayed_response', {'message': 'Started Training your model'})
    print("Emitted delayed response")
    socketio.sleep(5)
    socketio.emit('delayed_response', {'message': 'Finished Training your model'})
    print("Emitted delayed response AGAIN")

if __name__ == '__main__':
    # app.run(
    #     host='0.0.0.0', port=5000,
    #     ssl_context=('server.crt', 'server.key'),
    #     debug=True
    # )
    socketio.run(app,
        host='0.0.0.0', port=5000,
        ssl_context=('server.crt', 'server.key'),
        debug=True
    )
        

