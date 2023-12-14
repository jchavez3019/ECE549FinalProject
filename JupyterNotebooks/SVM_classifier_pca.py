import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from time import time
import pickle

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people #lfw dataset
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from sklearn.svm import SVC



class SVM_pca_classifier:
    def __init__(self, embeddings):

        ###### Load Data #############
        embeddings_arr = np.load(embeddings)

        file = open("./targetnames.txt")
        data = file.read()
        data2 = data.split("\n")
        file.close()

        self.target_names = data2
        print(f"len target names {len(self.target_names)}")
        
        # Extract X_train, y_train, X_test, and y_test directly
        self.X_train_pca, self.y_train_pca, self.X_test_pca, self.y_test_pca = [embeddings_arr[f'arr_{i}'] for i in range(4)]
        # # Print the shapes 
        print(self.X_train_pca.shape)
        print(self.y_train_pca.shape)
        print(self.X_test_pca.shape)
        print(self.y_test_pca.shape)

        self.y_train_pca = self.y_train_pca.ravel().astype('uint32')
        self.y_test_pca = self.y_test_pca.ravel().astype('uint32')

        # ####### Target Names ##############
        # with open(targetnames_file, "r") as file:
        #     names = [line.strip() for line in file]
        # # Convert the list of names to a NumPy array
        # y_train = np.array(names).reshape(-1, 1)
        # print("names array shape ", y_train.shape)

        # n_test = X_train_pca.shape[0] - y_train.shape[0]

    def load_svm(self, load_model_path: str):
        with open(load_model_path, 'rb') as f:
            self.svm_classifier = pickle.load(f)

    def train_svm(self, save_model_path: str = None):
        # Train a support vector machine (SVM) classifier
        self.svm_classifier = SVC(kernel='rbf', class_weight='balanced')
        print("Starting svm_classifier fit")
        start = time()
        self.svm_classifier.fit(self.X_train_pca, self.y_train_pca)
        end = time()
        print(f"Ended training in {end-start} s")

        if save_model_path is not None:
            print(f"Saving SVM model to path {save_model_path}")
            with open(save_model_path, 'wb') as file:
                pickle.dump(self.svm_classifier, file)

        ##### improve model using gridsearch #####
        # param_grid = {'C': [1, 5, 10,50], 'gamma': [0.0001, 0.0005, 0.001, 0.005]}
        # grid = GridSearchCV(svm_classifier, param_grid)
        # grid.fit(self.X_train_pca, self.y_train_pca) 

        # # Get the best hyperparameters
        # best_params = grid.best_params_
        # print("Best Hyperparameters:", best_params)

        # print(grid.best_params_)
        # model = grid.best_estimator_
        

    def predict_svm(self, save_class_report_path: str = None):
        print("Starting svm_classifer predict")
        start = time()
        yfit = self.svm_classifier.predict(self.X_test_pca)
        end = time()
        print(f"Ended prediction in {end-start} s")
        print(f"Shape of yfit: {yfit.shape}")
        print(f"Number of classes {len(self.target_names)}")

        print(confusion_matrix(self.y_test_pca, yfit, labels=range(len(self.target_names))))
        if save_class_report_path is None:
            print(classification_report(self.y_test_pca, yfit, labels=range(len(self.target_names)), target_names=self.target_names))
        else:
            report = classification_report(self.y_test_pca, yfit, labels=range(len(self.target_names)), target_names=self.target_names, output_dict=True)
            df = pd.DataFrame(report).transpose()
            df.to_csv(save_class_report_path)

        # Evaluate the model
        accuracy = accuracy_score(self.y_test_pca, yfit)
        print("Test Accuracy:", accuracy)


def main():
    embeddings_file = "lfw-deepfunneled-pca-embeddings-50.npz"
    svm_object = SVM_pca_classifier(embeddings_file)
    svm_object.train_svm('test_svm_model.pkl')
    # svm_object.load_svm('test_svm_model.pkl')
    svm_object.predict_svm('test_svm_model_report.csv')

if __name__ == "__main__":
    main()




