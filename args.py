import numpy as np


class BasicConfig:

    basedir = './'

    dtype = 'float32'

    nbatch = 3
    nepochs = 15
    lr_alpha = 0.01
    rank = 2
    indim = 56
    gpuid = -1
    inchan = 1

    n_c = { 'V1':6, 'V2':6 }

    standardize = True