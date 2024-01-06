import re

import numpy as np
import keras

from dataset_pre_process import split_test_train, DatasetPreProcess
from project_config_multi import Dataset


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, activity_labels, impairment_labels, pose_dir=None, rgb_dir=None,
                 **kwargs):
        'Initialization'

        self.list_IDs = list_IDs
        self.activity_labels = activity_labels
        self.impairment_labels = impairment_labels

        self.pose_dir = pose_dir
        self.rgb_dir = rgb_dir

        self.dim = kwargs['pose_dim']
        self.rgb_dim = kwargs['rgb_dim']
        self.batch_size = kwargs['batch_size']
        self.n_activity_classes = kwargs['n_activity_classes']
        self.n_impairment_classes = kwargs['n_impairment_classes']
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
        # R = np.empty((self.batch_size, *self.rgb_dim))
        y_a = np.empty((self.batch_size), dtype=int)
        y_i = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            # Store sample
            # print('ID: ', ID)
            subject = re.search("(S.+?)A", ID).group(1) + '/'
            activity = re.search("[0-9](A.+?)I", ID).group(1) + '/'
            impairment = re.search("[0-9](I.+?)R", ID).group(1) + '/'
            # print('subject, activity, impariment: ', subject, activity, impairment)
            pose_data = np.loadtxt(self.pose_dir + '/' + subject + activity + impairment + ID + '_kinect.txt')
            # print(self.rgb_dir)
            # R[i, ] = np.load(self.rgb_dir + '/' + ID + '_rgb.npy')
            # print(R[i].shape)
            # rgb_data = self._dpp.rgb_dim(rgb_data)
            # print('shape of kinect data: ', pose_data.shape)
            X[i, ] = self._dpp.dataset_pre_process(pose_data)
            # print('ID: ', ID)

            # Store activity
            y_a[i] = self.activity_labels[ID]

            # Store impairment
            y_i[i] = self.impairment_labels[ID]

        y_a = keras.utils.to_categorical(y_a, num_classes=self.n_activity_classes)
        y_i = keras.utils.to_categorical(y_i, num_classes=self.n_impairment_classes)

        y = np.concatenate((y_a, y_i), axis=1)
        # print('final label: ', y.shape)

        # return R, y
        return X, y


if __name__ == '__main__':

    config = Dataset()

    partition, activity_labels, impairment_labels = split_test_train(config.dataset_dir)

    # print('partition', partition)
    # print('activity_labels', activity_labels)
    # print('impairment_labels', impairment_labels)

    train_generator = DataGenerator(list_IDs=partition['train'],
                                    activity_labels=activity_labels,
                                    impairment_labels=impairment_labels,
                                    pose_dir=config.dataset_dir,
                                    # rgb_dir=config.dump_dir_rgb,
                                    **config.params)

    for i in range(980):
        print('train_generator', train_generator.__getitem__(0)[0].shape)
