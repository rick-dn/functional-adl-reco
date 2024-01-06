import re

import numpy as np
import keras

from dataset_pre_process import split_test_train_single, DatasetPreProcess
from project_config_single import Dataset


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, pose_dir=None, rgb_dir=None,
                 **kwargs):
        'Initialization'

        self.list_IDs = list_IDs
        self.labels = labels

        self.pose_dir = pose_dir
        self.rgb_dir = rgb_dir

        self.dim = kwargs['pose_dim']
        self.batch_size = kwargs['batch_size']
        self.shuffle = kwargs['shuffle']

        self._dpp = DatasetPreProcess(**kwargs)

        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty(self.batch_size, dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            # Store sample
            subject = re.search("(S.+?)A", ID).group(1) + '/'
            activity = re.search("[0-9](A.+?)I", ID).group(1) + '/'
            impairment = re.search("[0-9](I.+?)R", ID).group(1) + '/'
            # print('subject, activity, impariment: ', subject, activity, impairment)
            pose_data = np.loadtxt(self.pose_dir + subject + activity + impairment + ID + '_kinect.txt')
            # print('shape of kinect data: ', pose_data.shape)
            X[i, ] = self._dpp.dataset_pre_process(pose_data)
            # print('ID: ', ID, self.labels[ID])

            # Store activity
            y[i] = self.labels[ID]

        return X, y


if __name__ == '__main__':

    config = Dataset()

    partition, labels = split_test_train_single(config.dataset_dir)

    # print('partition', partition)
    # print('activity_labels', activity_labels)
    # print('impairment_labels', impairment_labels)

    train_generator = DataGenerator(list_IDs=partition['train'],
                                    labels=labels,
                                    pose_dir=config.dataset_dir,
                                    **config.params)

    print('train_generator', train_generator.__getitem__(0)[1].shape)
