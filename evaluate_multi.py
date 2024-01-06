import numpy as np
import tensorflow as tf
from keras_layer_normalization import LayerNormalization
from keras_self_attention import SeqSelfAttention
from sklearn.metrics import accuracy_score
from keras import models

from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns

from data_generator_multi import DataGenerator
from dataset_pre_process import split_test_train
from net_fv import NetFV_Temporal, NetFV_Spatial
from project_config_multi import Dataset

config = Dataset()
print(config.working_dir)


#  test train split
partition, activity_labels, impairment_labels = split_test_train(config.dataset_dir)


def numpy_multi_label_accuracy(y_tru, y_prd):
    y_true_a = np.argmax(y_tru[:, :10], axis=1)
    y_pred_a = np.argmax(y_prd[:, :10], axis=1)
    print('activity score: ', accuracy_score(y_true_a, y_pred_a))
    m_a = y_true_a == y_pred_a

    y_true_i = np.argmax(y_tru[:, 10:], axis=1)
    y_pred_i = np.argmax(y_prd[:, 10:], axis=1)
    print('impariment score: ', accuracy_score(y_true_i, y_pred_i))
    m_i = y_true_i == y_pred_i

    # print('m_a', m_a)
    # print('m_i', m_i)
    # print(m_a * m_i)

    print('combined accuracy: ', np.count_nonzero(m_a * m_i) / np.size(m_a))

    return y_true_a, y_true_i, y_pred_a,  y_pred_i


def show_confusion_matrix(y_true, y_pred):

    cm_map = confusion_matrix(y_true, y_pred)
    plt.clf()
    plt.figure(figsize=(35, 35))
    plt.title('Confusion matrix of the classifier')
    sns.heatmap(cm_map, annot=True, fmt='g')
    plt.xlabel('Predicted')
    plt.xticks()
    plt.ylabel('True')
    plt.yticks()
    plt.ioff()
    plt.show()
    plt.draw()

    return plt


def main():

    custom_objects = {'SeqSelfAttention': SeqSelfAttention,
                      'LayerNormalization': LayerNormalization,
                      'NetFV_Temporal': NetFV_Temporal,
                      'NetFV_Spatial': NetFV_Spatial,
                      'tf': tf,
                      }

    validation_generator = DataGenerator(list_IDs=partition['validation'],
                                         activity_labels=activity_labels,
                                         impairment_labels=impairment_labels,
                                         pose_dir=config.dataset_dir,
                                         **config.params)

    model = models.load_model(config.loading_file, custom_objects=custom_objects)
    print('model loaded')
    print(model.summary())

    pred_probs = model.predict_generator(validation_generator)
    print('Prediction done')
    # print('pred_probs ', pred_probs.shape)

    y_true = np.zeros(shape=pred_probs.shape)
    for index in range(1760 // 16):
        y_true[index * 16:index * 16 + 16] = validation_generator.__getitem__(index)[1]
    print('labels generated')

    y_true_a, y_true_i, y_pred_a,  y_pred_i = numpy_multi_label_accuracy(y_true, pred_probs)

    show_confusion_matrix(y_true_a, y_pred_a)
    show_confusion_matrix(y_true_i, y_pred_i)


if __name__ == '__main__':
    main()
