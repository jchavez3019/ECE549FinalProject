############## USING GPUS in Parallel ##################
import numpy as np
import cupy as cp
from time import time
from cuml import SVC
from cuml.model_selection import GridSearchCV
from cuml.metrics import accuracy_score


class SVM_pca_classifier_CUML:
    def __init__(self, embeddings):

        ###### Load Data #############
        embeddings_arr = cp.load(embeddings)
        # Extract X_train, y_train, X_test, and y_test directly
        self.X_train_pca, self.y_train_pca, self.X_test_pca, self.y_test_pca = [embeddings_arr[f'arr_{i}'] for i in range(4)]
        # Print the shapes
        print(self.X_train_pca.shape)
        print(self.y_train_pca.shape)
        print(self.X_test_pca.shape)
        print(self.y_test_pca.shape)

        self.y_train_pca = self.y_train_pca.ravel()
        self.y_test_pca = self.y_test_pca.ravel()

    def train_svm(self):
        # Train a support vector machine (SVM) classifier
        start = time()
        print(f"Start of fit")
        svm_classifier = SVC(kernel='rbf', class_weight='balanced')
        svm_classifier.fit(self.X_train_pca, self.y_train_pca)
        end = time()
        print(f"End of fit {end - start} s")

        ##### improve model using gridsearch #####
        # param_grid = {'C': [1, 5, 10,50], 'gamma': [0.0001, 0.0005, 0.001, 0.005]}
        # grid = GridSearchCV(svm_classifier, param_grid)
        # grid.fit(self.X_train_pca, self.y_train_pca)

        # # Get the best hyperparameters
        # best_params = grid.best_params_
        # print("Best Hyperparameters:", best_params)

        # print(grid.best_params_)
        # model = grid.best_estimator_
        start = time()
        print(f"Start of predict")
        yfit = svm_classifier.predict(cp.asarray(self.X_test_pca))
        end = time()
        print(f"End of predict {end - start} s")
        # Evaluate the model
        accuracy = accuracy_score(self.y_test_pca, yfit)
        print("Test Accuracy:", accuracy)


def main():
    embeddings_file = "./lfw-deepfunneled-pca-embeddings-50.npz"
    svm_object = SVM_pca_classifier_CUML(embeddings_file)
    svm_object.train_svm()


if __name__ == "__main__":
    main()