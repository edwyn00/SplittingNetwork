import json
import numpy as np
import random
from  NetworkCONFIG import *

class DatasetManager:

    def __init__(self):

        self.element_shape = ELEMENT_SHAPE
        self.batch_size = BATCH_SIZE
        self.x_tensorDim = (BATCH_SIZE, ELEMENT_SHAPE[0], ELEMENT_SHAPE[1], ELEMENT_SHAPE[2], ELEMENT_SHAPE[3])
        self.y_tensorDim = (BATCH_SIZE, 2)

        #self.new_epoch()


    def next_batch(self):
        zeros_or_ones = random.randint(0,1)
        if zeros_or_ones == 0:
            x_train = np.zeros(self.x_tensorDim)
            y_train = np.zeros(self.y_tensorDim)
            y_train[:,0] += 1
            return x_train, y_train
        else:
            x_train = np.ones(self.x_tensorDim)
            y_train = np.zeros(self.y_tensorDim)
            y_train[:,1] += 1
            return x_train, y_train

    #################
    #   TODO        #
    #################

    def test_batch(self, dim):
        return


    #################
    #   TODO        #
    #################

    def new_epoch(self):
        return
