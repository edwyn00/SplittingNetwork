from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from DatasetManager import *

from NetworkCONFIG import *


class C3D(tf.keras.layers.Layer):
    def __init__(self, filter_dims, kernel_sizes, paddings):
        super(C3D, self).__init__()
        self.blocks = []
        for filters, kernel_size, padding in zip(filter_dims, kernel_sizes, paddings):

            c3d_layer = tf.keras.layers.Conv3D(filters, kernel_size, padding='valid')
            self.blocks.append(c3d_layer)

            batch_norm = tf.keras.layers.BatchNormalization()
            self.blocks.append(batch_norm)

            relu = tf.keras.layers.ReLU()
            self.blocks.append(relu)

    def call(self, input_features):
        x = self.blocks[0](input_features)
        for i in range(1, len(self.blocks)):
             x = self.blocks[i](x)
        return x


class resNet(tf.keras.layers.Layer):

    def __init__(self, filter_dims, kernel_sizes, paddings):
        super(resNet, self).__init__()
        self.blocks = []
        self.blocks.append(tf.keras.layers.Reshape(RESNET_INIT, input_shape=C3D_FINAL_SHAPE))
        for filters, kernel_size, padding in zip(filter_dims, kernel_sizes, paddings):
            self.residual_block(filters, kernel_size, padding)


    def residual_block(self, filters, kernel_size, padding):
        momentum    = RESIDUAL_BLOCK_MOMENTUM
        activation  = RESIDUAL_BLOCK_ACTIVATION

        res = tf.keras.layers.Conv2D(filters=filters[0], kernel_size=kernel_size, padding=padding)
        self.blocks.append(res)

        res = tf.keras.layers.Activation(activation=activation)
        self.blocks.append(res)

        res = tf.keras.layers.BatchNormalization(momentum=momentum)
        self.blocks.append(res)

        res = tf.keras.layers.Conv2D(filters=filters[1], kernel_size=kernel_size, padding=padding)
        self.blocks.append(res)

        res = tf.keras.layers.BatchNormalization(momentum=momentum)
        self.blocks.append(res)

        res = tf.keras.layers.Add()
        self.blocks.append(res)

    def call(self, x):
        mem = np.copy(x)
        for i in range(0, len(self.blocks)):
            if (i)%6==0:
                x = self.blocks[i]([x,mem])
                mem = np.copy(x)
            else:
                x = self.blocks[i](x)
        return x


class binaryClassificator(tf.keras.layers.Layer):
    def __init__(self, layer_dims):
        super(binaryClassificator, self).__init__()
        self.blocks = []
        self.blocks.append(tf.keras.layers.Flatten())
        for layer_dim in layer_dims:
            x = tf.keras.layers.Dense(units=layer_dim, activation=tf.nn.relu)
            self.blocks.append(x)

    def call(self, x):
        for i in range(0, len(self.blocks)):
             x = self.blocks[i](x)
        return x

class splittingNetwork(tf.keras.layers.Layer):
    def __init__(self):
        super(splittingNetwork, self).__init__()

        self.c3d                    = C3D                  (CONV3D_FILTERS_SHAPE, CONV3D_KERNELS_SHAPE, CONV3D_PADDINGS)
        self.resnet                 = resNet               (RESNET_FILTERS_SHAPE, RESNET_KERNELS_SHAPE, RESNET_PADDINGS)
        self.binary_classificator   = binaryClassificator  (DENSE_SHAPE)

    def call(self, input_features):
        x = self.c3d(input_features)
        x = self.resnet(x)
        x = self.binary_classificator(x)
        return x

def loss(model, x_train, y_train):
    intermidiate_loss = tf.square(tf.subtract(model(x_train), y_train))
    reconstruction_error = tf.reduce_mean(intermidiate_loss)
    return reconstruction_error


def train(loss, model, opt, x_train=None, y_train=None):
    with tf.GradientTape() as tape:
        gradients = tape.gradient(loss(model, x_train, y_train), model.trainable_variables)
        gradient_variables = zip(gradients, model.trainable_variables)
        opt.apply_gradients(gradient_variables)

def main():

    #########################
    #   DATASET MANAGER     #
    #########################

    dataset = DatasetManager()

    #########################
    #   NETWORK             #
    #########################
    splitting_network = splittingNetwork()

    #########################
    #   OPTIMIZER           #
    #########################
    opt = tf.optimizers.Adam(learning_rate=LEARNING_RATE)

    #########################
    #   TENSORBOARD         #
    #########################
    writer = tf.summary.create_file_writer('tmp')
    tf.summary.trace_on(graph=True, profiler=True)

    with writer.as_default():
        tf.summary.trace_export(name="my_test", step=0, profiler_outdir='tmp')

        #########################
        #   EPOCH LOOP          #
        #########################
        for epoch in range(EPOCHS):
            dataset.new_epoch()
            print("\nEpoch number", epoch)

            #########################
            #   MAIN STEP LOOP      #
            #########################
            for step in range(STEPS_PER_EPOCH):

                x_train, y_train = dataset.next_batch()
                train(loss, splitting_network, opt, x_train=x_train, y_train=y_train)
                loss_values = loss(splitting_network, x_train, y_train)

                tf.summary.scalar('loss', loss_values, step=step + epoch*STEPS_PER_EPOCH)

                if step%10==0:
                    print("Loss for the step", step, "is", loss_values)
