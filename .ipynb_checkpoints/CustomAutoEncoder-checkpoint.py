import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
import warnings as w
w.filterwarnings('ignore')


def AutoencoderNoisy(inp_shape, output_dim, target_dim,
                     hidden_layers_encoder = [], dropout_encoder = [], 
                     hidden_layers_dencoder = [], dropout_dencoder = [], 
                     noise_level = 0.005, momentum=0.99, use_complex = False, use_noise = True):
    inp = layers.Input(shape= (inp_shape,))
    
    encoded = layers.BatchNormalization(momentum = momentum)(inp)
    if use_noise:
        encoded = layers.GaussianNoise(noise_level)(encoded)
    encoded = layers.Dense(output_dim, activation = layers.LeakyReLU())(encoded)
    
    encoded2 = layers.Dropout(0.4)(encoded)
    for idx, units in enumerate(hidden_layers_encoder):
        encoded2 = layers.Dense(units, activation = "relu")(encoded2)
        encoded2 = layers.BatchNormalization()(encoded2)
        encoded2 = layers.Dropout(dropout_encoder[idx])(encoded2)
    encoded2 = layers.Dense(output_dim, activation = layers.LeakyReLU())(encoded2)
    
    next_inp = encoded
    if use_complex:
        next_inp = encoded2
    decoded = layers.Dropout(0.2)(next_inp)
    decoded = layers.BatchNormalization()(decoded)
    for idx, units in enumerate(hidden_layers_dencoder):
        decoded = layers.Dense(units, activation = "relu")(decoded)
        decoded = layers.BatchNormalization()(decoded)
        decoded = layers.Dropout(dropout_dencoder[idx])(decoded)
    decoded = layers.Dense(output_dim, activation = 'relu',name='decoded')(decoded)
    
    x = layers.Dense(32,activation='relu')(decoded)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32,activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)    
    x = layers.Dense(target_dim,activation='sigmoid',name='label_output')(x)
    
    autoencoder = Model(inputs = inp, outputs = [decoded, x])
    encoder = Model(inputs = inp, outputs = encoded)
    encoder2 = Model(inputs = inp, outputs = encoded2)
    
    autoencoder.compile(optimizer=Adam(0.005),loss={'decoded':'mse','label_output':'binary_crossentropy'})
    return autoencoder, encoder, encoder2

a, b, c = AutoencoderNoisy(train_features.shape[-1], encoded_features_shape, target.shape[-1], use_complex = False)