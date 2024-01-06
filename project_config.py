import os


class Config:

    params = {'batch_size': 16,
              'shuffle': True,
              'epochs': 200,
              'steps_per_epoch': 980,  # 980,  # 980
              'learning_rate': 0.01,
              'momentum': 0.9,
              'decay': 1e-6,  # 1e-3
              # 'loss': 'categorical_crossentropy',
              # 'loss': 'binary_crossentropy',
              'loss': 'sparse_categorical_crossentropy',
              'data_aug': True,
              'reg': 1e-5,
              'dropout': 0.5,
              'activation': 'relu'}


class Dataset(Config):

    dataset_dir = '/data/localdrive/datasets/dataset_post_processed/kinect_data'
    dataset_dir_rgb = '/data/localdrive/datasets/dataset_post_processed/rgb_data'
    working_dir = '/data/localdrive/datasets/'
    dump_dir = working_dir + '/dump/rgb/'
    dump_dir_pose = working_dir + '/dump/pose/'

    # checkpoint dir
    checkpoint_dir = working_dir + '/initial_run_multi/'
    activity_checkpoint = checkpoint_dir + '/{epoch:03d}_{val_activity_op_accuracy:0.3f}.hd5'
    impairment_checkpoint = checkpoint_dir + '/{epoch:03d}_{val_impairment_op_accuracy:0.3f}.hd5'
    model_save_dir = checkpoint_dir + '/final_model.json'
    loading_file = working_dir + '/'

    Config.params['frames'] = 600
    Config.params['persons'] = 1
    Config.params['joints_per_person'] = 20
    Config.params['coord_per_joint'] = 3
    # Config.params['raw_coord'] = 12
    Config.params['depth_norm'] = 53000
    Config.params['pose_dim'] = (Config.params['frames'],
                                 Config.params['joints_per_person'] *
                                 Config.params['persons'] *
                                 Config.params['coord_per_joint'])
    Config.params['raw_rgb_dim'] = (Config.params['frames'], 1080, 1920, 3)
    Config.params['rgb_dim'] = (Config.params['frames'], 224, 224, 3)

    Config.params['n_activity_classes'] = 10
    Config.params['n_impairment_classes'] = 8

    Config.params['n_classes'] = Config.params['n_activity_classes'] + \
                                 Config.params['n_impairment_classes']
