from keras.models import Model
from keras.layers import Input
from keras.layers import add
import tensorflow as tf
from keras import layers
from keras.layers import Activation, Dense, Flatten, Dropout
from keras.layers import Conv1D, AveragePooling1D, MaxPooling1D, Lambda
from keras.layers import GlobalAveragePooling1D, Reshape, multiply
from keras.regularizers import l2, l1
from keras.layers import BatchNormalization
# from keras_layer_normalization import LayerNormalization
from keras_multi_head import MultiHeadAttention
from keras.utils import plot_model
from keras import backend as K

# from loupe_keras import NetFV
# from net_vlad import NetVLAD
# from attn_class import AttentionWithContext
from keras_self_attention import SeqSelfAttention, ScaledDotProductAttention, SeqWeightedAttention


def TCN_resnet(
        n_classes,
        input_tensor,
        gap=1,
        dropout=0.0,
        name='name',
        kernel_regularizer=l1(1.e-4),
        activation="relu"):
    if K.image_data_format() == 'tf':
        ROW_AXIS = 1
        CHANNEL_AXIS = 2
    else:
        ROW_AXIS = 2
        CHANNEL_AXIS = 1

    config = [
        [(1, 8, 64)],
        [(1, 8, 64)],
        [(1, 8, 64)],
        [(2, 8, 128)],
        [(1, 8, 128)],
        [(1, 8, 128)],
        [(2, 8, 256)],
        [(1, 8, 256)],
        [(1, 8, 256)]
    ]
    initial_stride = 1
    initial_filter_dim = 8
    initial_num = 64

    # input = Input(batch_shape=(batch_size, *dim))
    # model = input

    model = Conv1D(initial_num,
                   initial_filter_dim,
                   strides=initial_stride,
                   padding="same",
                   kernel_initializer="he_normal",
                   kernel_regularizer=kernel_regularizer)(input_tensor)

    for depth in range(0, len(config)):
        for stride, filter_dim, num in config[depth]:
            bn = BatchNormalization(axis=CHANNEL_AXIS)(model)
            relu = Activation(activation)(bn)
            dr = Dropout(dropout)(relu)
            res = Conv1D(num,
                         filter_dim,
                         strides=stride,
                         padding="same",
                         kernel_initializer="he_normal",
                         kernel_regularizer=kernel_regularizer)(dr)

            res_shape = K.int_shape(res)
            model_shape = K.int_shape(model)
            if res_shape[CHANNEL_AXIS] != model_shape[CHANNEL_AXIS]:
                model = Conv1D(num,
                               1,
                               strides=stride,
                               padding="same",
                               kernel_initializer="he_normal",
                               kernel_regularizer=kernel_regularizer)(model)

            model = add([model, res])

    bn = BatchNormalization(axis=CHANNEL_AXIS)(model)
    model = Activation(activation)(bn)

    if gap:
        # pool_window_shape = K.int_shape(model)
        flatten = GlobalAveragePooling1D()(model)
        # gap = AveragePooling1D(pool_window_shape[ROW_AXIS],
        #                        strides=1)(model)
        # flatten = Flatten()(gap)
    else:
        flatten = Flatten()(model)
    dense = Dense(units=n_classes,
                  activation="softmax",
                  kernel_initializer="he_normal",
                  name=name)(flatten)

    # model = Model(inputs=input, outputs=dense)
    return dense


def TCN_resnet_multi(n_activity_classes=10,
                     n_impairment_classes=8,
                     batch_size=16,
                     dim=(600, 60),
                     gap=True,
                     kernel_regularizer=None,
                     activation=l2(1e-5)):

    x = Input(batch_shape=(batch_size, *dim))

    activity_stream = TCN_resnet(n_classes=n_activity_classes, input_tensor=x,
                                 kernel_regularizer=kernel_regularizer, gap=gap,
                                 activation=activation, name='activity_op')

    impairment_stream = TCN_resnet(n_classes=n_impairment_classes, input_tensor=x,
                                   kernel_regularizer=kernel_regularizer, gap=gap,
                                   activation=activation, name='impairment_op')

    model = Model(inputs=x, outputs=[activity_stream, impairment_stream])

    return model





def tcn_resnet(n_classes, batch_size, dim, gap=1,
               kernel_regularizer=l2(1.e-4),
               activation="relu"):
    if K.image_dim_ordering() == 'tf':
        ROW_AXIS = 1
        CHANNEL_AXIS = 2
    else:
        ROW_AXIS = 2
        CHANNEL_AXIS = 1

    # ORIGINAL
    config = [
        [(1, 8, 64)],
        [(1, 8, 64)],
        [(1, 8, 64)],
        [(2, 8, 128)],
        [(1, 8, 128)],
        [(1, 8, 128)],
        [(2, 8, 256)],
        [(1, 8, 256)],
        [(1, 8, 256)]
    ]
    initial_stride = 1
    initial_filter_dim = 8
    initial_num = 64

    input = Input(batch_shape=(batch_size, *dim))
    model = input
    model = Conv1D(initial_num,
                   initial_filter_dim,
                   strides=initial_stride,
                   padding="same",
                   kernel_initializer="he_normal",
                   kernel_regularizer=kernel_regularizer)(model)

    model = BatchNormalization(axis=CHANNEL_AXIS)(model)
    model = Activation(activation)(model)

    for depth in range(0, len(config)):
        for stride, filter_dim, num in config[depth]:

            res = Conv1D(num,
                         filter_dim,
                         strides=stride,
                         padding="same",
                         kernel_initializer="he_normal",
                         kernel_regularizer=kernel_regularizer)(model)

            res = BatchNormalization(axis=CHANNEL_AXIS)(res)
            res = Activation(activation)(res)

            res_shape = K.int_shape(res)
            model_shape = K.int_shape(model)
            if res_shape[CHANNEL_AXIS] != model_shape[CHANNEL_AXIS]:
                model = Conv1D(num,
                               1,
                               strides=stride,
                               padding="same",
                               kernel_initializer="he_normal",
                               kernel_regularizer=kernel_regularizer)(model)
                # model = Activation(activation)(model)
            model = add([model, res])
            # model = SeqSelfAttention()(model)

    bn = BatchNormalization(axis=CHANNEL_AXIS)(model)
    model = Activation(activation)(bn)

    print('model before attn', model)
    # model = SeqSelfAttention()(model)
    print('seq self attn')

    if gap:
        pool_window_shape = K.int_shape(model)
        gap = AveragePooling1D(pool_window_shape[ROW_AXIS],
                               strides=1)(model)
        # gap = MaxPooling1D(pool_window_shape[ROW_AXIS],
        #                        strides=1)(model)
        flatten = Flatten()(gap)
    else:
        flatten = Flatten()(model)
    dense = Dense(units=n_classes,
                  activation="softmax",
                  kernel_regularizer=kernel_regularizer,
                  kernel_initializer="glorot_normal")(flatten)

    # model = Lambda(lambda x: K.reshape(x, (-1, 256)))(model)

    # print('net vlad')
    # dense = NetFV(feature_size=256, max_samples=22, cluster_size=64, output_dim=60)(model)
    # dense = NetVLAD(feature_size=256, max_samples=22, cluster_size=64, output_dim=60)(model)

    model = Model(inputs=input, outputs=dense)

    print(model.summary())
    return model


def tcn_resnet_spatial(
        dim,
        dropout=0.0,
        kernel_regularizer=l1(1.e-4),
        activation="relu"):
    if K.image_dim_ordering() == 'tf':
        ROW_AXIS = 1
        CHANNEL_AXIS = 2
    else:
        ROW_AXIS = 2
        CHANNEL_AXIS = 1

    config = [
        [(1, 8, 64)],
        [(1, 8, 64)],
        [(1, 8, 64)],
        [(2, 8, 128)],
        [(1, 8, 128)],
        [(1, 8, 128)],
        [(2, 8, 256)],
        [(1, 8, 256)],
        [(1, 8, 256)]
    ]
    initial_stride = 1
    initial_filter_dim = 8
    initial_num = 64

    input = Input(shape=dim)
    model = input

    model = Conv1D(initial_num,
                   initial_filter_dim,
                   strides=initial_stride,
                   padding="same",
                   kernel_initializer="he_normal",
                   kernel_regularizer=kernel_regularizer)(model)

    for depth in range(0, len(config)):
        for stride, filter_dim, num in config[depth]:
            bn = BatchNormalization(axis=CHANNEL_AXIS)(model)
            relu = Activation(activation)(bn)
            dr = Dropout(dropout)(relu)
            res = Conv1D(num,
                         filter_dim,
                         strides=stride,
                         padding="same",
                         kernel_initializer="he_normal",
                         kernel_regularizer=kernel_regularizer)(dr)

            res_shape = K.int_shape(res)
            model_shape = K.int_shape(model)
            if res_shape[CHANNEL_AXIS] != model_shape[CHANNEL_AXIS]:
                model = Conv1D(num,
                               1,
                               strides=stride,
                               padding="same",
                               kernel_initializer="he_normal",
                               kernel_regularizer=kernel_regularizer)(model)

            model = add([model, res])

    bn = BatchNormalization(axis=CHANNEL_AXIS)(model)
    model = Activation(activation)(bn)

    model = Model(inputs=input, outputs=model)
    return model


def tcn_resnet_temporal(
        dim,
        dropout=0.0,
        kernel_regularizer=l1(1.e-4),
        activation="relu"):
    if K.image_dim_ordering() == 'tf':
        ROW_AXIS = 1
        CHANNEL_AXIS = 2
    else:
        ROW_AXIS = 2
        CHANNEL_AXIS = 1

    config = [
        [(1, 8, 64)],
        [(1, 8, 64)],
        [(1, 8, 64)],
        [(2, 8, 128)],
        [(1, 8, 128)],
        [(1, 8, 128)],
        [(2, 8, 256)],
        [(1, 8, 256)],
        [(1, 8, 256)]
    ]
    initial_stride = 1
    initial_filter_dim = 8
    initial_num = 64

    input = Input(shape=dim)
    model = input

    model = Conv1D(initial_num,
                   initial_filter_dim,
                   strides=initial_stride,
                   padding="same",
                   kernel_initializer="he_normal",
                   kernel_regularizer=kernel_regularizer)(model)

    for depth in range(0, len(config)):
        for stride, filter_dim, num in config[depth]:
            bn = BatchNormalization(axis=CHANNEL_AXIS)(model)
            relu = Activation(activation)(bn)
            dr = Dropout(dropout)(relu)
            res = Conv1D(num,
                         filter_dim,
                         strides=stride,
                         padding="same",
                         kernel_initializer="he_normal",
                         kernel_regularizer=kernel_regularizer)(dr)

            res_shape = K.int_shape(res)
            model_shape = K.int_shape(model)
            if res_shape[CHANNEL_AXIS] != model_shape[CHANNEL_AXIS]:
                model = Conv1D(num,
                               1,
                               strides=stride,
                               padding="same",
                               kernel_initializer="he_normal",
                               kernel_regularizer=kernel_regularizer)(model)

            model = add([model, res])

    bn = BatchNormalization(axis=CHANNEL_AXIS)(model)
    model = Activation(activation)(bn)

    model = Model(inputs=input, outputs=model)
    return model


def tcn_resnet_1(dim, gap=1,
                 kernel_regularizer=l2(1.e-4),
                 activation="relu"):
    if K.image_dim_ordering() == 'tf':
        ROW_AXIS = 1
        CHANNEL_AXIS = 2
    else:
        ROW_AXIS = 2
        CHANNEL_AXIS = 1

    # ORIGINAL
    config = [
        [(1, 8, 64)],
        [(1, 8, 64)],
        [(1, 8, 64)],
        # [(2, 8, 128)],
        # [(1, 8, 128)],
        # [(1, 8, 128)],
        # [(2, 8, 256)],
        # [(1, 8, 256)],
        # [(1, 8, 256)]
    ]
    initial_stride = 1
    initial_filter_dim = 8
    initial_num = 64

    x = Input(shape=dim)
    model = Conv1D(initial_num,
                   initial_filter_dim,
                   strides=initial_stride,
                   padding="same",
                   kernel_initializer="he_normal",
                   kernel_regularizer=kernel_regularizer)(x)

    model = BatchNormalization(axis=CHANNEL_AXIS)(model)
    # model = LayerNormalization()(model)
    model = Activation(activation)(model)

    for depth in range(0, len(config)):
        for stride, filter_dim, num in config[depth]:

            res = Conv1D(num,
                         filter_dim,
                         strides=stride,
                         padding="same",
                         kernel_initializer="he_normal",
                         kernel_regularizer=kernel_regularizer)(model)

            res = BatchNormalization(axis=CHANNEL_AXIS)(res)
            # res = LayerNormalization()(res)
            res = Activation(activation)(res)

            res_shape = K.int_shape(res)
            model_shape = K.int_shape(model)
            if res_shape[CHANNEL_AXIS] != model_shape[CHANNEL_AXIS]:
                model = Conv1D(num,
                               1,
                               strides=stride,
                               padding="same",
                               kernel_initializer="he_normal",
                               kernel_regularizer=kernel_regularizer)(model)
                # model = LayerNormalization()(model)
                # model = Activation(activation)(model)
            model = add([model, res])
            # model = MultiHeadAttention(head_num=4, kernel_regularizer=kernel_regularizer)([model, res, model])
            # model = LayerNormalization()(model)
            bn = BatchNormalization(axis=CHANNEL_AXIS)(model)
            # model = SeqSelfAttention()(model)

    bn = BatchNormalization(axis=CHANNEL_AXIS)(model)
    # bn = LayerNormalization()(model)
    model = Activation(activation)(bn)

    model = Model(inputs=x, outputs=model)

    # print(model.summary())
    return model


def tcn_resnet_2(n_classes, batch_size, dim, gap=1,
                 kernel_regularizer=l2(1.e-4),
                 activation="relu"):
    if K.image_dim_ordering() == 'tf':
        ROW_AXIS = 1
        CHANNEL_AXIS = 2
    else:
        ROW_AXIS = 2
        CHANNEL_AXIS = 1

    # ORIGINAL
    config = [
        # [(1, 8, 64)],
        # [(1, 8, 64)],
        # [(1, 8, 64)],
        [(2, 8, 128)],
        [(1, 8, 128)],
        [(1, 8, 128)],
        [(2, 8, 256)],
        [(1, 8, 256)],
        [(1, 8, 256)]
    ]
    initial_stride = 1
    initial_filter_dim = 8
    initial_num = 64

    x = Input(batch_shape=(batch_size, *dim))
    model = Conv1D(initial_num,
                   initial_filter_dim,
                   strides=initial_stride,
                   padding="same",
                   kernel_initializer="he_normal",
                   kernel_regularizer=kernel_regularizer)(x)

    # model = LayerNormalization()(model)
    model = BatchNormalization(axis=CHANNEL_AXIS)(model)
    model = Activation(activation)(model)

    for depth in range(0, len(config)):
        for stride, filter_dim, num in config[depth]:

            res = Conv1D(num,
                         filter_dim,
                         strides=stride,
                         padding="same",
                         kernel_initializer="he_normal",
                         kernel_regularizer=kernel_regularizer)(model)

            # res = LayerNormalization()(res)
            res = BatchNormalization(axis=CHANNEL_AXIS)(res)
            res = Activation(activation)(res)

            res_shape = K.int_shape(res)
            model_shape = K.int_shape(model)
            if res_shape[CHANNEL_AXIS] != model_shape[CHANNEL_AXIS]:
                model = Conv1D(num,
                               1,
                               strides=stride,
                               padding="same",
                               kernel_initializer="he_normal",
                               kernel_regularizer=kernel_regularizer)(model)
                # model = LayerNormalization()(model)
                # model = Activation(activation)(model)
            model = add([model, res])
            # model = MultiHeadAttention(head_num=4, kernel_regularizer=kernel_regularizer)([model, res, model])
            # model = LayerNormalization()(model)
            model = BatchNormalization(axis=CHANNEL_AXIS)(model)
            # model = SeqSelfAttention()(model)

    bn = BatchNormalization(axis=CHANNEL_AXIS)(model)
    # bn = LayerNormalization()(model)
    model = Activation(activation)(bn)

    model = Model(inputs=x, outputs=model)

    # print(model.summary())
    return model


def tcn_resnet_enc(batch_size, n_classes, dim=None,
                    gap=1, kernel_regularizer=None, activation=None):

    if K.image_dim_ordering() == 'tf':
        ROW_AXIS = 1
        CHANNEL_AXIS = 2
    else:
        ROW_AXIS = 2
        CHANNEL_AXIS = 1

    x = Input(batch_shape=(batch_size, dim[0], dim[1]))

    print('x: ', x)

    y_temporal = Lambda(lambda i: K.permute_dimensions(i, (0, 2, 1)))(x)
    y_temporal = tcn_resnet_temporal(dim=(150, 300),
                                     kernel_regularizer=kernel_regularizer,
                                     activation=activation)(y_temporal)
    y_temporal = Lambda(lambda i: K.permute_dimensions(i, (0, 2, 1)))(y_temporal)
    print('y_temporal: ', y_temporal)

    y_spatial = Lambda(lambda i: K.reshape(i, (batch_size * dim[0], 50, 3)))(x)
    print('y_spatial after lambda 1: ', y_spatial)
    y_spatial = tcn_resnet_1(dim=(25, 3),
                     kernel_regularizer=kernel_regularizer,
                     activation=activation)(y_spatial)

    print('y_spatial before lambda 2: ', y_spatial)
    y_spatial = Lambda(lambda i: K.reshape(i, (batch_size, dim[0], 3200)))(y_spatial)
    print('y_spatial after lambda 2: ', y_spatial)
    y_spatial = tcn_resnet_2(
        n_classes=n_classes,
        batch_size=batch_size,
        dim=(dim[0], 3200),
        gap=1,
        kernel_regularizer=kernel_regularizer,
        activation=activation)(y_spatial)
    print('y_spatial final: ', y_spatial)

    y = layers.Multiply()([y_spatial, y_temporal])
    print('final y: ', y)

    # model = SeqSelfAttention()(model)
    # print('seq self attn')

    # model = MultiHeadAttention(head_num=4, kernel_regularizer=kernel_regularizer)(model)
    # model = LayerNormalization()(model)
    # model = SeqSelfAttention(attention_activation='sigmoid', kernel_regularizer=kernel_regularizer)(model)
    # model = SeqWeightedAttention()(model)
    # bn = BatchNormalization(axis=CHANNEL_AXIS)(model)
    # print('y_spatial after attn: ', y_spatial)
    # model = layers.Bidirectional(layers.LSTM(32, kernel_initializer='he_normal',
    #                                        kernel_regularizer=kernel_regularizer,
    #                                        return_sequences=True))(model)
    # bn = BatchNormalization(axis=CHANNEL_AXIS)(model)
    # model = LayerNormalization()(model)

    # model = Lambda(lambda x: K.reshape(x, (-1, 256)))(model)

    # print('net vlad')
    # dense = NetFV(feature_size=256, max_samples=22, cluster_size=64, output_dim=60)(model)
    # dense = NetVLAD(feature_size=256, max_samples=22, cluster_size=64, output_dim=60)(model)

    if gap:
        print('if seeing this then GAP is on')
        pool_window_shape = K.int_shape(y_spatial)
        y = AveragePooling1D(pool_window_shape[ROW_AXIS],
                                 strides=1)(y)
        y = Flatten()(y)

    y = Dense(units=n_classes,
              activation="softmax",
              kernel_regularizer=kernel_regularizer,
              kernel_initializer="he_normal")(y)

    model = Model(inputs=x, outputs=y)
    # model.summary()

    return model

def L1(x):
  out = []
  for i in range(16):
    out.append(x[i]/K.sum(K.abs(x[i])))
  out = tf.reshape(out, (16, -1))
  return out

def tcn_resnet_spatial_temporal(batch_size, n_classes, dim=None,
                    gap=1, kernel_regularizer=None, activation=None):

    if K.image_dim_ordering() == 'tf':
        ROW_AXIS = 1
        CHANNEL_AXIS = 2
    else:
        ROW_AXIS = 2
        CHANNEL_AXIS = 1

    x = Input(batch_shape=(batch_size, dim[0], dim[1]))

    print('x: ', x)

    y_spatial = Lambda(lambda i: K.reshape(i, (batch_size * dim[0], 50, 3)))(x)
    print('y_spatial after lambda 1: ', y_spatial)
    y_spatial = tcn_resnet_1(dim=(50, 3),
                     kernel_regularizer=kernel_regularizer,
                     activation=activation)(y_spatial)

    print('y_spatial before lambda 2: ', y_spatial)
    y_spatial = Lambda(lambda i: K.reshape(i, (batch_size, dim[0], 3200)))(y_spatial)
    print('y_spatial after lambda 2: ', y_spatial)
    y_spatial = tcn_resnet_2(
        n_classes=n_classes,
        batch_size=batch_size,
        dim=(dim[0], 3200),
        gap=1,
        kernel_regularizer=kernel_regularizer,
        activation=activation)(y_spatial)
    print('y_spatial final: ', y_spatial)


    # if gap:
    #     print('if seeing this then GAP is on')
    #     pool_window_shape = K.int_shape(y_spatial)
    #     y_spatial = AveragePooling1D(pool_window_shape[ROW_AXIS],
    #                              strides=1)(y_spatial)
    #     y_spatial = Flatten()(y_spatial)

    # y_spatial = Dense(units=n_classes,
    #           activation="softmax",
    #           kernel_regularizer=kernel_regularizer,
    #           kernel_initializer="he_normal")(y_spatial)
    print('using netfv cs = 32: ')
    y_spatial = NetFV(feature_size=256, max_samples=75, cluster_size=32, output_dim=60)(y_spatial)
    y_spatial = layers.Activation('softmax')(y_spatial)
    print('y_spatial: ', y_spatial)

    y_temporal = Lambda(lambda i: K.permute_dimensions(i, (0, 2, 1)))(x)
    # y_temporal = Lambda(lambda i: K.reshape(i, (batch_size * dim[1], 50, 6)))(y_temporal)
    print('y_temporal after lambda 1: ', y_temporal)
    y_temporal = tcn_resnet_1(dim=(150, 300),
                     kernel_regularizer=kernel_regularizer,
                     activation=activation)(y_temporal)

    print('y_temporal before lambda 2: ', y_temporal)
    # y_temporal = Lambda(lambda i: K.reshape(i, (batch_size, dim[1], 3200)))(y_temporal)
    y_temporal = Lambda(lambda i: K.permute_dimensions(i, (0, 2, 1)))(y_temporal)
    print('y_temporal after lambda 2: ', y_temporal)
    y_temporal = tcn_resnet_2(
        n_classes=n_classes,
        batch_size=batch_size,
        dim=(dim[1], 150),
        gap=1,
        kernel_regularizer=kernel_regularizer,
        activation=activation)(y_temporal)
    print('y_temporal final: ', y_temporal)

    # y_temporal = Lambda(lambda i: K.permute_dimensions(i, (0, 2, 1)))(y_temporal)
    # if gap:
    #     print('if seeing this then GAP is on')
    #     pool_window_shape = K.int_shape(y_temporal)
    #     y_temporal = AveragePooling1D(pool_window_shape[ROW_AXIS],
    #                              strides=1)(y_temporal)
    #     y_temporal = Flatten()(y_temporal)

    # y_temporal = Dense(units=n_classes,
    #           activation="softmax",
    #           kernel_regularizer=kernel_regularizer,
    #           kernel_initializer="he_normal")(y_temporal)
    print('using netfv cs = 32: ')
    y_temporal = NetFV(feature_size=256, max_samples=16, cluster_size=32, output_dim=60)(y_temporal)
    y_temporal = layers.Activation('softmax')(y_temporal)
    print('y_temporal: ', y_temporal)

    # y = layers.Multiply()([y_spatial, y_temporal])
    y = layers.multiply([y_spatial, y_temporal])
    y = Lambda(L1)(y)
    print('final y: ', y)


    model = Model(inputs=x, outputs=y)
    # model.summary()

    return model


def main():
    n_classes = 60
    reg = l1(1.e-4)
    dropout = 0.8
    activation = "relu"

    # model = tcn_resnet(n_classes, dim=(190, 20), gap=1,
    #                    kernel_regularizer=reg,
    #                    activation=activation)

    # model = tcn_resnet_enc(batch_size=8, n_classes=n_classes, dim=(300, 150),
    #                        kernel_regularizer=reg, activation=activation)

    model = tcn_resnet_spatial_temporal(batch_size=16, n_classes=n_classes, dim=(300, 150),
                                        kernel_regularizer=reg, activation=activation)

    # print(model.summary())
    # plot_model(model, to_file='tcn_resnet.png')


if __name__ == '__main__':
    K.clear_session()
    main()
