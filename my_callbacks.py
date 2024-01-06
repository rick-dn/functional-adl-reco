import os

import cv2
from keras.callbacks import Callback
import matplotlib
# import matplotlib.patches as mpatches
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import sklearn
import seaborn as sns
# import itertools
import numpy as np
import time
import matplotlib.pyplot as plt

# matplotlib.use('Agg')


class MasterCallback(Callback):

    def __init__(self, validation_generator, batch_size, results_dir=None, load_file=None):
        self.validation_generator = validation_generator
        self.batch_size = batch_size
        self.results_dir = results_dir
        self.load_file = load_file
        self._acc = 0
        self._time = time.time()
        # plt.ion()
        if not os.path.isdir(self.results_dir):
            print(self.results_dir)
            raise FileNotFoundError

        super(MasterCallback, self).__init__()

    def on_train_begin(self, logs={}):

        self._time = time.time()
        # pred_proba, acc, y_true, y_pred = self.validate()
        # self.show_confusion_matrix(y_true, y_pred)
        if self.load_file:
            _, self._acc, _, _ = self.validate()
            print('starting validation acc: {:.2f}'.format(self._acc*100))

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs={}):

        self.model.save(self.results_dir + '/model_weights.h5')

        # if logs['accuracy'] > 0.90:
        if True:

            pred_probs, acc, y_true, y_pred = self.validate()
            # self.show_confusion_matrix(y_true, y_pred)
            logs['val_acc'] = acc

            if acc > self._acc:

                print('validation accuracy improved from {:.2f} to {:.2f}'.format(self._acc*100, acc*100,))
                self._acc = acc

                if os.path.isdir(self.results_dir):

                    self.model.save(self.results_dir + '/model_{:03d}_{:.2f}.h5'.format(epoch, acc * 100))
                    np.savez(self.results_dir + '/preds_{:03d}_{:.2f}'.format(epoch, acc * 100),
                             pred_probs, acc, y_pred, y_true)

                    try:

                        cm_map = self.show_confusion_matrix(y_true, y_pred)
                        cm_map.savefig(self.results_dir + '/confusion_matrix.png', dpi=300)
                        cm_map.close()

                    except Exception:
                        print('exception in saving confusion matrix')
                    finally:
                        print('continuing..')

                else:
                    print('No working dir given, not saving data')

            else:
                print('validation accuracy {:.2f} did not improve from {:.2f}'.format(acc * 100, self._acc * 100))

        print('time elapsed: {:.2f}'.format(time.time() - self._time))
        self._time = time.time()

    # def on_epoch_end(self, epoch, logs=None):
    #
    #     if logs['accuracy'] > 0.9:
    #         loss, val_acc = self.model.evaluate_generator(self.validation_generator)
    #         print('val loss: {:.2f}, val acc: {:.2f}'.format(loss, val_acc))
    #
    #         if val_acc > self._acc:
    #             print('val acc improved from {:.2f} to {:.2f}'.format(self._acc*100, val_acc*100))
    #             self._acc = val_acc
    #             self.model.save(self.results_dir + '/model_{:03d}_{:.2f}.hdf5'.format(epoch, val_acc*100))
    #
    #         else:
    #             print('val acc: {:.2f}, did not improve from: {:.2f}'.format(val_acc*100, self._acc*100))

    def on_train_end(self, logs=None):

        pred_probs, acc, y_true, y_pred = self.validate()
        print('validation accuracy: {:.2f}'.format(acc*100))

        # if acc > 0.89:
        # np.savez(self.results_dir + '/preds_test', pred_probs, acc, y_pred, y_true)
        # cm_map = self.show_confusion_matrix(y_true, y_pred)
        # cm_map.savefig(self.results_dir + '/confusion_matrix.png', dpi=300)
        # cm_map.show()
        # cm_map.close()

        return pred_probs, acc, y_pred, y_true

    def validate(self):

        pred_probs = self.model.predict_generator(self.validation_generator)
        # print('pred_probs shape: ', pred_probs.shape)

        y_true = np.zeros(shape=(pred_probs.shape[0]))
        for index in range(0, pred_probs.shape[0] // self.batch_size):
            y_true[index * self.batch_size:index * self.batch_size + self.batch_size] = self.validation_generator.__getitem__(index)[1]

        y_pred = np.argmax(pred_probs, axis=1)
        print(y_true)
        # print('print shapes', y_pred.shape, y_true.shape)

        acc = accuracy_score(y_true, y_pred)

        return pred_probs, acc, y_true, y_pred

    @staticmethod
    def show_confusion_matrix(y_true, y_pred):

        cm_map = confusion_matrix(y_true, y_pred)
        # plt.clf()
        plt.figure(figsize=(35, 35))
        plt.title('Confusion matrix of the classifier')
        sns.heatmap(cm_map, annot=True)
        plt.xlabel('Predicted')
        plt.xticks()
        plt.ylabel('True')
        plt.yticks()
        # plt.ioff()
        # plt.show()
        # plt.draw()

        return plt