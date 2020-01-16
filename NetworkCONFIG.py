##########################
#   NETWORK PARAMS       #
##########################

LEARNING_RATE   = 0.005
STEPS_PER_EPOCH = 50
EPOCHS          = 5
BATCH_SIZE      = 1
#ELEMENT_SHAPE   = [300, 225, 225, 3]
ELEMENT_SHAPE   = [30, 20, 20, 3]
C3D_FINAL_SHAPE = (24, 14, 14, 256)
RESNET_INIT = (24*14, 14, 256)

##########################
#       Conv3D           #
##########################

BASE = 64
CONV3D_FILTERS_SHAPE = [BASE     , BASE*2,    BASE*4]
CONV3D_KERNELS_SHAPE = [(3, 3, 3), (3, 3, 3), (3, 3, 3)]
CONV3D_PADDINGS      = ['same',    'same',    'same'   ]

##########################
#       ResNet           #
##########################

NUM_OF_BLOCKS               = 15
RESIDUAL_BLOCK_MOMENTUM     = 0.8
RESIDUAL_BLOCK_ACTIVATION   = 'relu'

RESNET_FILTERS_SHAPE = [[256,256]] * NUM_OF_BLOCKS
RESNET_KERNELS_SHAPE = [3        ] * NUM_OF_BLOCKS
RESNET_PADDINGS      = ['same'   ] * NUM_OF_BLOCKS

##########################
#       Dense            #
##########################

DENSE_SHAPE          = [64, 32, 2]
