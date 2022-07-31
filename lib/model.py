import numpy as np
import tensorflow as tf

"""Model definitions."""


def iid2model(qpm, policy):
    """Convert IID policy to a model."""
    a0sz = qpm[2] - qpm[0] + 1
    a1sz = qpm[2] - qpm[1] + 1
    inp = tf.keras.Input(shape=(1,))
    out = tf.keras.layers.Dense(a0sz + a1sz)(inp)
    model = tf.keras.Model(inputs=inp, outputs=out)

    kernel = np.zeros((1, a0sz + a1sz), np.float32)
    bias = np.zeros((a0sz + a1sz,), np.float32)

    kernel[0, a0sz:] = 1
    bias[(qpm[1] - qpm[0]):a0sz] = policy

    model.set_weights([kernel, bias])
    return model


def mlp(insz, outsz, nhidden, nlayers):
    """Define an mlp model."""
    inp = tf.keras.Input(shape=(insz,))
    _y = inp
    for _ in range(nlayers):
        _y = tf.keras.layers.Dense(nhidden, kernel_initializer='he_uniform')(_y)
        _y = tf.keras.layers.LeakyReLU(alpha=0.2)(_y)
    out = tf.keras.layers.Dense(outsz, kernel_initializer='he_uniform')(_y)
    return tf.keras.Model(inputs=inp, outputs=out)
