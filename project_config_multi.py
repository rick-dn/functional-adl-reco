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
              'loss': 'binary_crossentropy',
              # 'loss': 'sparse_categorical_crossentropy',
              'metrics': 'categorical_accuracy',
              # 'metrics': 'accuracy',
              'data_aug': True,
              'reg': 1e-5,
              'dropout': 0.5,
              'activation': 'relu'}


class Dataset(Config):

    dataset_dir = '/data/localdrive/datasets/dataset_post_processed/kinect_data'
    dataset_dir_rgb = '/data/localdrive/datasets/dataset_post_processed/rgb_data'
    working_dir = '/data/localdrive/datasets/'
    # dump_dir_rgb = working_dir + '/dump/rgb_112/'
    dump_dir_rgb = working_dir + '/dump/rgb/'
    dump_dir_pose = working_dir + '/dump/pose/'

    # checkpoint dir
    checkpoint_dir = working_dir + '/tcn_resnet_enc_fv_16_16_16_09/'
    checkpoint = checkpoint_dir + '/{epoch:03d}_{val_categorical_accuracy:0.3f}.hd5'
    model_save_dir = checkpoint_dir + '/final_model.json'
    # loading_file = working_dir + '/multi_label_encodings_no_fv/012_0.658.hd5'
    loading_file = working_dir + '/multi_label_encodings/016_0.714.hd5'
    # loading_file = working_dir + '/c3d_multi_label_05th_sep/009_0.546.hd5'
    Config.params['frames'] = 600
    Config.params['rgb_frames'] = 20
    Config.params['persons'] = 1
    Config.params['joints_per_person'] = 20
    Config.params['coord_per_joint'] = 3
    # Config.params['raw_coord'] = 12
    Config.params['depth_norm'] = 53000
    Config.params['pose_dim'] = (Config.params['frames'],
                                 Config.params['joints_per_person'] *
                                 Config.params['persons'] *
                                 Config.params['coord_per_joint'])
    Config.params['raw_rgb_dim'] = (Config.params['frames'], 640, 480, 3)
    Config.params['rgb_dim'] = (Config.params['rgb_frames'], 299, 299, 3)

    Config.params['n_activity_classes'] = 10
    Config.params['n_impairment_classes'] = 8

    Config.params['n_classes'] = Config.params['n_activity_classes'] + \
                                 Config.params['n_impairment_classes']
    Config.params['loading_file'] = loading_file
