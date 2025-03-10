__authors__ = ['1674393', '1671219', '1666675', '1672973']
__group__ = '13'

import numpy as np
import math
import operator
from scipy.spatial.distance import cdist


class KNN:
    def __init__(self, train_data, labels):
        self._init_train(train_data)
        self.labels = np.array(labels)
        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_train(self, train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        train_data = train_data.astype(float)
        train_data = np.reshape(train_data, [train_data.shape[0], train_data.shape[1]*train_data.shape[2]])
        self.train_data = train_data

    def get_k_neighbours(self, test_data, k):
        """
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        self.test_data = np.reshape(test_data, [test_data.shape[0], test_data.shape[1]*test_data.shape[2]])
        Mdist = cdist(self.test_data, self.train_data)
        
        idx = np.argsort(Mdist, axis=1)[:, :k]
        self.neighbors = self.labels[idx]

        
    def get_class(self):
        """
        Get the class by maximum voting
        :return: 1 array of Nx1 elements. For each of the rows in self.neighbors gets the most voted value
                (i.e. the class at which that row belongs)
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        most_common = []
        
        for row in self.neighbors:
            unique_labels, label_counts = np.unique(row, return_counts=True)
            
            indexes = sorted(range(len(unique_labels)), key=lambda i: np.where(row == unique_labels[i])[0][0])
            unique_labels = unique_labels[indexes]
            label_counts = label_counts[indexes]
            
            most_common_label_idx = np.argmax(label_counts, axis=0)
            most_common_labels = unique_labels[most_common_label_idx]
            most_common.append(most_common_labels)

        return most_common

    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the output form get_class a Nx1 vector with the predicted shape for each test image
        """

        self.get_k_neighbours(test_data, k)
        return self.get_class()
