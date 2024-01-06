import os
import time
import numpy as np
import re

import pathlib
from pathlib import Path
import cv2 as cv
import pylab as pl
from imutils.video import FileVideoStream, FPS, count_frames
# print(pl.rcParams['backend'])
from tqdm import tqdm
from multiprocessing import Pool

# pl.rcParams['backend'] = 'TkAgg'  # 'WXAgg'
from matplotlib import pyplot as plt

from project_config_multi import Dataset


class DatasetPreProcess:

    def __init__(self, **kwargs):

        self.no_of_frames = kwargs['frames']
        self.no_of_rgb_frames = kwargs['rgb_frames']
        self.joints_per_person = kwargs['joints_per_person']
        self.coord_per_joint = kwargs['coord_per_joint']
        # self.raw_coord = kwargs['raw_coord']
        self.persons = kwargs['persons']
        self.rgb_dim = kwargs['rgb_dim']
        self.depth_norm = kwargs['depth_norm']

    def dataset_pre_process(self, x):

        pose_data = np.zeros((self.no_of_frames, self.joints_per_person * self.coord_per_joint))

        if x.shape[0] == 0:
            x = np.zeros((self.no_of_frames, self.joints_per_person * self.coord_per_joint))

        if x.shape[0] <= self.no_of_frames:

            pose_data[:x.shape[0]] = x.reshape(x.shape[0], -1)

        else:
            index = np.asarray(np.floor(np.linspace(0, x.shape[0] - 1, self.no_of_frames)), dtype=int)
            pose_data = x[index].reshape(self.no_of_frames, -1)

        return pose_data

    @staticmethod
    def get_rgb_data(rgb_file):

        # read video file
        # print('rgb_file: ', rgb_file)
        cap = cv.VideoCapture(rgb_file)

        frame_info = [int(cap.get(cv.CAP_PROP_FRAME_COUNT)),
                      int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
                      int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)),
                      int(cap.get(cv.CAP_PROP_CHANNEL))]
        # print('number of frames, frame width, height: ', frame_info)

        frame_counter = 0
        rgb_data = []

        # Read until video is completed
        while cap.isOpened():

            # Capture frame-by-frame
            ret, frame = cap.read()

            if ret:

                # Display the resulting frame
                # cv.imshow('Frame', frame)

                frame_counter += 1

                rgb_data.append(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

                # Press Q on keyboard to  exit
                # if cv.waitKey(25) & 0xFF == ord('q'):
                #     break

            # Break the loop
            else:
                break

        # When everything done, release the video capture object
        cap.release()

        # Closes all the frames
        # cv.destroyAllWindows()

        # assert
        assert frame_info[0] == frame_counter

        return np.asarray(rgb_data), frame_info

    def dataset_pre_process_rgb(self, x):

        rgb_data = np.zeros((self.no_of_rgb_frames, *x.shape[1:]))
        # print('init shape', rgb_data.shape)

        if x.shape[0] <= self.no_of_rgb_frames:

            rgb_data[:x.shape[0]] = x

        else:
            index = np.asarray(np.floor(np.linspace(0, x.shape[0] - 1, self.no_of_rgb_frames)), dtype=int)
            rgb_data = x[index]

        # print('rgb_data shape before resize: ', rgb_data.shape)

        # display rgb
        # for frame in rgb_data:
        #     # draw joints matplotlib
        #     # plt.scatter(joint[:, 0], joint[:, 1])
        #     # Display the resulting frame
        #     plt.imshow(frame)
        #     plt.show()
        #     # exit()
        # exit()

        rgb_data_resized = []

        for index, image in enumerate(rgb_data):

            # resize to fit nn
            image = cv.resize(image, (self.rgb_dim[1], self.rgb_dim[2]))

            rgb_data_resized.append(image)

        return np.asarray(rgb_data_resized, dtype=np.int32)
        # return rgb_data_resized


def split_test_train(source):

    partition = {}
    activity_labels = {}
    impairment_labels = {}
    train_ids = []
    val_ids = []

    train_counter = 0
    val_counter = 0

    x_max = 0
    frames = []
    # getting ID from text file
    for index, filename in enumerate(pathlib.Path(source).rglob('*_kinect.txt')):

        # x = np.loadtxt(filename).shape[0]
        # frames.append(x)
        # if x > x_max:
        #     x_max = x

        # Split 1
        train_subs = [0, 1, 2, 3, 4, 6, 8]
        val_subs = [5, 7, 9]

        ID = re.search("S0[0-9][0-9]A0[0-9][0-9]I0[0-9][0-9]R0[0-9][0-9]", filename.name).group()
        subject = int(re.search("S(.+?)A", ID).group(1)) - 1
        activity = int(re.search("A(.+?)I", ID).group(1)) - 1
        impairment = int(re.search("I(.+?)R", ID).group(1)) - 1

        # if index < 100:
        #     print('ID, subject, activity, impairment: ', ID, subject, activity, impairment)

        if subject in train_subs:
            train_ids.append(ID)
            train_counter += 1
        else:
            val_ids.append(ID)
            val_counter += 1

        activity_labels[ID] = activity
        impairment_labels[ID] = impairment

    partition['train'] = train_ids
    partition['validation'] = val_ids

    # print('maximum frame: ', x_max)
    print('total, train counter, val counter: ', train_counter + val_counter, train_counter, val_counter)

    # plt.hist(np.asarray(frames), bins=50)
    # plt.show()

    return partition, activity_labels, impairment_labels


def split_test_train_single(source):

    partition = {}
    labels = {}
    label_number = -1
    label_map = {}
    train_ids = []
    val_ids = []

    train_counter = 0
    val_counter = 0

    x_max = 0
    frames = []
    # getting ID from text file
    for index, filename in enumerate(pathlib.Path(source).rglob('*_kinect.txt')):

        # x = np.loadtxt(filename).shape[0]
        # frames.append(x)
        # if x > x_max:
        #     x_max = x

        # Split 1
        train_subs = [0, 1, 2, 3, 4, 6, 8]
        val_subs = [5, 7, 9]

        ID = re.search("S0[0-9][0-9]A0[0-9][0-9]I0[0-9][0-9]R0[0-9][0-9]", filename.name).group()
        subject = int(re.search("S(.+?)A", ID).group(1)) - 1
        activity = str(re.search("[0-9](A.+?)R", ID).group(1))
        # impairment = int(re.search("I(.+?)R", ID).group(1))

        # if index < 100:
        #     print('ID, subject, activity, impairment: ', ID, subject, activity, impairment)

        if subject in train_subs:
            train_ids.append(ID)
            train_counter += 1
        else:
            val_ids.append(ID)
            val_counter += 1

        if activity not in label_map.keys():
            label_number +=1
            label_map[activity] = label_number

        labels[ID] = label_map[activity]

    partition['train'] = train_ids
    partition['validation'] = val_ids

    print('label_map', label_map)
    # print('maximum frame: ', x_max)
    print('total, train counter, val counter: ', train_counter + val_counter, train_counter, val_counter)

    # plt.hist(np.asarray(frames), bins=50)
    # plt.show()

    return partition, labels


def main():
    config = Dataset()

    split_test_train(config.dataset_dir_rgb)
    # split_test_train_single(config.dataset_dir_rgb)

    if not os.path.isdir(config.dump_dir_rgb):
        os.mkdir(config.dump_dir_rgb)

    dataset_pre_process = DatasetPreProcess(**config.params)

    # pose_data = msr_pre_process.get_pose_data(open(config.dataset_dir + '/a13_s09_e02_skeleton.txt', 'r'))
    # print('pose data', pose_data)

    for index, filename in tqdm(enumerate(pathlib.Path(config.dataset_dir_rgb).rglob('*_rgb.avi'))):
        # print('file being processed: ', index, filename.as_posix())
        rgb_data, frame_info = dataset_pre_process.get_rgb_data(filename.as_posix())
        # print('rgb_data_shape after file read', rgb_data.shape)

        # display rgb
        # for frame in rgb_data[:1]:
        #     # draw joints matplotlib
        #     # plt.scatter(joint[:, 0], joint[:, 1])
        #     # Display the resulting frame
        #     plt.imshow(frame)
        #     plt.show()
        #     exit()
        # exit()

        rgb_data = dataset_pre_process.dataset_pre_process_rgb(rgb_data)
        # print('processed rgb_data_shape', rgb_data.shape)

        # display rgb
        # for frame in rgb_data:
        #     # draw joints matplotlib
        #     # plt.scatter(joint[:, 0], joint[:, 1])
        #     # Display the resulting frame
        #     print(frame)
        #     plt.imshow(frame)
        #     plt.show()
        #     # exit()
        # exit()

        # normalize
        rgb_data = rgb_data / 127.5
        rgb_data = rgb_data - 1

        # print('rgb data shape: ', rgb_data.shape)
        # print('dump file name: ', config.dump_dir_rgb + os.path.basename(filename.name).split(".")[0])
        # exit()
        # np.save(config.dump_dir_rgb + os.path.basename(filename.name).split(".")[0], rgb_data)


if __name__ == '__main__':
    main()