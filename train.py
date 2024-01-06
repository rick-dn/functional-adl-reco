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

from data_generator import DataGenerator
from dataset_pre_process import split_test_train
from project_config import Dataset
from tcn_resnet import tcn_resnet_spatial_temporal, TCN_resnet, TCN_resnet_multi

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
# reg = l2(config.params['reg'])
reg = l1(1e-5)
# reg = l1_l2(config.params['reg'])

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
model = TCN_resnet_multi(n_activity_classes=config.params['n_activity_classes'],
                         n_impairment_classes=config.params['n_impairment_classes'],
                         batch_size=config.params['batch_size'],
                         dim=config.params['pose_dim'],
                         gap=True,
                         kernel_regularizer=reg,
                         activation=activation
                         )

print(model.summary())
model.compile(optimizer=optimizer, loss={'activity_op': loss, 'impairment_op': loss}, metrics=['accuracy'])
# model.load_weights(config.params['loading_file'])

try:

    if not os.path.isdir(config.checkpoint_dir):
        os.mkdir(config.checkpoint_dir)

    # callbacks
    activity_checkpoint = ModelCheckpoint(filepath=config.activity_checkpoint,
                                          monitor='val_activity_op_accuracy',
                                          verbose=1,
                                          save_weights_only=False,
                                          save_best_only=True, mode='max')

    impairment_checkpoint = ModelCheckpoint(filepath=config.impairment_checkpoint,
                                            monitor='val_impairment_op_accuracy',
                                            verbose=1,
                                            save_weights_only=False,
                                            save_best_only=True, mode='max')

    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.1,
                                  patience=10,
                                  verbose=1,
                                  mode='auto',
                                  cooldown=3,
                                  min_lr=0.0000001)

    # my_callback = MasterCallback(validation_generator, config.params['batch_size'], results_dir=config.checkpoint_dir)

    # callbacks_list = [my_callback, reduce_lr]
    callbacks_list = [activity_checkpoint, impairment_checkpoint, reduce_lr]

    model.fit_generator(generator=train_generator,
                        steps_per_epoch=config.params['steps_per_epoch'],
                        epochs=config.params['epochs'],
                        callbacks=callbacks_list,
                        validation_data=validation_generator,
                        use_multiprocessing=False,
                        workers=10,
                        verbose=1)

finally:

    print('highest acc activity: {:.2f}'.format(np.max(model.history.history.get('activity_op_accuracy'))))
    print('highest acc activity epoch: {}'.format(np.argmax(model.history.history.get('activity_op_accuracy'))))
    print('highest acc impairment: {:.2f}'.format(np.max(model.history.history.get('impairment_op_accuracy'))))
    print('highest acc impairment epoch: {}'.format(np.argmax(model.history.history.get('impairment_op_accuracy'))))
    print('highest acc val activity: {:.2f}'.format(np.max(model.history.history.get('val_activity_op_accuracy'))))
    print('highest acc val activity epoch: {}'.format(np.argmax(model.history.history.get('val_activity_op_accuracy'))))
    print('highest acc val impairment: {:.2f}'.format(np.max(model.history.history.get('val_impairment_op_accuracy'))))
    print('highest acc val impairment epoch: {}'.format(np.argmax(model.history.history.get('val_impairment_op_accuracy'))))

    # serialize model to JSON
    model_json = model.to_json()
    with open(config.model_save_dir, "w") as json_file:
        json_file.write(model_json)


