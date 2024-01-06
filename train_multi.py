from matplotlib import pyplot as plt
import numpy as np
from sys import exit
import os

import tensorflow as tf
import keras
 
from keras.regularizers import l2, l1, l1_l2
from keras.optimizers import RMSprop, SGD, Adam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import keras.backend as K
from sklearn.metrics import accuracy_score

from data_generator_multi import DataGenerator
from dataset_pre_process import split_test_train
from project_config_multi import Dataset
from tcn_resnet_multi import tcn_resnet_spatial_enc, TCN_resnet, tcn_resnet_spatial_temporal

from my_callbacks import MasterCallback


# def numpy_multi_label_accuracy(y_tru, y_prd):
#     y_true_a = np.argmax(y_tru[:, :10], axis=1)
#     y_pred_a = np.argmax(y_prd[:, :10], axis=1)
#     print('activity score: ', accuracy_score(y_true_a, y_pred_a))
#     m_a = y_true_a == y_pred_a
#
#     y_true_i = np.argmax(y_tru[:, 10:], axis=1)
#     y_pred_i = np.argmax(y_prd[:, 10:], axis=1)
#     print('impariment score: ', accuracy_score(y_true_i, y_pred_i))
#     m_i = y_true_i == y_pred_i
#
#     # print('m_a', m_a)
#     # print('m_i', m_i)
#     # print(m_a * m_i)
#
#     # y_true_c = np.stack((y_true_a, y_true_i), axis=1)
#     # y_pred_c = np.stack((y_pred_a, y_pred_i), axis=1)
#     # print('y_true_c', y_true_c)
#
#     print('combined accuracy: ', np.count_nonzero(m_a * m_i) / np.size(m_a))
#
#     # return np.count_nonzero(m_a * m_i) / np.size(m_a)

K.clear_session()
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# print('tf, keras version:', tf.__version__, keras.__version__)

# Parameters
config = Dataset()
print(config.working_dir)

# OPTIMIZER PARAMS
loss = config.params['loss']
lr = config.params['learning_rate']
momentum = config.params['momentum']
decay = config.params['decay']
activation = config.params['activation']
# optimizer = SGD(lr=lr, momentum=momentum, decay=decay, nesterov=True)
optimizer = SGD(lr=lr, momentum=momentum, decay=0.0, nesterov=True)
# optimizer = Adam(lr=lr, decay=decay)
# optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay)
# optimizer = RMSprop(lr=lr, decay=decay, epsilon=1e-08)
dropout = config.params['dropout']
# reg = l1(config.params['reg'])
reg = l2(config.params['reg'])
# reg = l1(1e-5)
# reg = l1_l2(config.params['reg'])
metrics = config.params['metrics']

#  test train split
partition, activity_labels, impairment_labels = split_test_train(config.dataset_dir)

# Generators
train_generator = DataGenerator(list_IDs=partition['train'],
                                activity_labels=activity_labels,
                                impairment_labels=impairment_labels,
                                pose_dir=config.dataset_dir,
                                **config.params)

validation_generator = DataGenerator(list_IDs=partition['validation'],
                                     activity_labels=activity_labels,
                                     impairment_labels=impairment_labels,
                                     pose_dir=config.dataset_dir,
                                     **config.params)

# models
# model = TCN_resnet(n_classes=config.params['n_classes'],
#                batch_size=config.params['batch_size'],
#                dim=config.params['pose_dim'],
#                gap=True,
#                kernel_regularizer=reg,
#                activation=activation
#                )

# model = tcn_resnet_spatial_enc(batch_size=config.params['batch_size'], n_classes=config.params['n_classes'],
#                                dim=config.params['pose_dim'], gap=True, kernel_regularizer=reg, activation=activation)

model = tcn_resnet_spatial_temporal(batch_size=config.params['batch_size'], n_classes=config.params['n_classes'],
                                    dim=config.params['pose_dim'],
                                    gap=True, kernel_regularizer=reg, activation=activation)

print(model.summary())
model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])
# model.load_weights(config.params['loading_file'])
# exit()

try:

    if not os.path.isdir(config.checkpoint_dir):
        os.mkdir(config.checkpoint_dir)

    # callbacks
    checkpoint = ModelCheckpoint(filepath=config.checkpoint,
                                 monitor='val_categorical_accuracy',
                                 verbose=1,
                                 save_weights_only=False,
                                 save_best_only=True, mode='max')

    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.1,
                                  patience=20,
                                  verbose=1,
                                  mode='auto',
                                  cooldown=3,
                                  min_lr=0.0000001)

    my_callback = MasterCallback(validation_generator, config.params['batch_size'], results_dir=config.checkpoint_dir)

    # callbacks_list = [my_callback, reduce_lr]
    callbacks_list = [checkpoint, reduce_lr]

    # model.load_weights(config.loading_file)
    # print('model loaded')
    #
    # pred_probs = model.predict_generator(validation_generator)
    # print('test loss, test acc: ', pred_probs.shape)
    #
    # y_true = np.zeros(shape=pred_probs.shape)
    # for index in range(1760//16):
    #     y_true[index * 16:index * 16 + 16] = validation_generator.__getitem__(index)[1]
    #
    # numpy_multi_label_accuracy(y_true, pred_probs)
    #
    # exit()

    model.fit_generator(generator=train_generator,
                        steps_per_epoch=config.params['steps_per_epoch'],
                        epochs=config.params['epochs'],
                        callbacks=callbacks_list,
                        validation_data=validation_generator,
                        use_multiprocessing=False,
                        workers=10,
                        verbose=1)

finally:

    print('highest acc: {:.2f}'.format(np.max(model.history.history.get('categorical_accuracy'))))
    print('highest acc: {}'.format(np.argmax(model.history.history.get('categorical_accuracy'))))
    print('highest acc val: {:.2f}'.format(np.max(model.history.history.get('val_categorical_accuracy'))))
    print('highest acc val epoch: {}'.format(np.argmax(model.history.history.get('val_categorical_accuracy'))))

    # serialize model to JSON
    model_json = model.to_json()
    with open(config.model_save_dir, "w") as json_file:
        json_file.write(model_json)


