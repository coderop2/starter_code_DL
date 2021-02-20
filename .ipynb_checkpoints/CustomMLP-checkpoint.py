import warnings as wr
import numpy as np
import pandas as pd
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import RMSprop, Adagrad, Adam
from tensorflow.python.keras.metrics import Metric as m
wr.filterwarnings('ignore')

def MLP(inp_shape, output_dims, output_activation, NNlayers = [], layer_dropouts = [], dense_regs = [regularizers.l1_l2(l1=1e-5, l2=1e-4)],
       learning_rate = 0.01, label_smoothing = 1e-2, momentum = 0.99):
    inp = layers.Input(shape=(inp_shape,))
    x = layers.BatchNormalization(momentum = 0.99)(inp)
    x = layers.Dense(inp_shape*1.5, activation = layers.LeakyReLU(), kernel_regularizer = dense_regs[0])(x)
    x = layers.Dropout(0.5)(x)
    for idx, units in enumerate(NNlayers):
        x = layers.Dense(units,activation = "relu", kernel_regularizer = dense_regs[idx + 1])(x)
        x = layers.BatchNormalization(momentum = 0.99)(x)
        x = layers.Dropout(layer_dropouts[idx])(x)
    x = layers.Dense(output_dims, activation = output_activation)(x)
    
    model = Model(inputs = inp, outputs = x)
    
    model.compile(metrics = [tf.keras.metrics.Accuracy(name="accuracy")],
                 optimizer = Adam(learning_rate),
                 loss = BinaryCrossentropy(label_smoothing = label_smoothing))
    
    return model

model = MLP(124, 1, 'sigmoid', [128], [0.2],[regularizers.l1_l2(l1=1e-5, l2=1e-4), None])