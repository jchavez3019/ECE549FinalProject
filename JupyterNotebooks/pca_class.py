import numpy as np
import matplotlib.pyplot as plt
from numpy import load, savez_compressed
import pandas as pd
from sklearn.decomposition import IncrementalPCA
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
from os import listdir
import re

class LFW_IncrementalPCA(IncrementalPCA):
    def __init__(self, dataset_paths: list[str], n_components: int, batch_size: int = 16, whiten: bool = True):
        super(LFW_IncrementalPCA, self).__init__( 
            n_components=n_components,
            batch_size = batch_size,
            whiten=whiten
        )
        # pca = IncrementalPCA(n_components=n_components,batch_size = batch_size)
        self.dataset_paths = dataset_paths

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
        return X_train_pca, Y_train_pca, X_test_pca, Y_test_pca
    
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

if __name__ == '__main__':

    """ Tests running the class on some local dataset files """

    import re

    n_components = 150
    dataset_npz_filenames = []
    reg_expr = '^lfw-deepfunneled-dataset_[0-9]{4}.npz'
    for file in listdir('./'):
        if re.search(reg_expr, file):
            dataset_npz_filenames.append(file)
    dataset_npz_filenames = sorted(dataset_npz_filenames)
    print(dataset_npz_filenames)
    # create an lfw_ipca instance and give the label of datasets and desired n_componentes
    lfw_ipca = LFW_IncrementalPCA(dataset_npz_filenames, n_components=n_components)
    
    # start the fit on the entire datasets given 
    lfw_ipca.start_fit(display_progress=True)

    # plot the embeddings in a 2d space
    # lfw_ipca.display_embedding_space()

    # save the embeddings as a compressed .npz file
    lfw_ipca.save_embeddings("lfw-deepfunneled-pca-embeddings-50.npz", display_progress=True)