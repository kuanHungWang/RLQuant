import tensorflow as tf
from tensorflow.keras import layers


def get_actor(n_hidden, N_OBSERVATION):
    inputs = layers.Input(shape=(N_OBSERVATION))
    x = inputs
    for l in n_hidden:
        x = layers.Dense(l, activation='relu')(x)
    x = layers.Dense(1, activation='tanh')(x)
    return tf.keras.Model(inputs, x)


def get_critic(n_hidden, N_OBSERVATION):
    inputs = layers.Input(shape=(N_OBSERVATION))
    x = inputs
    for l in n_hidden:
        x = layers.Dense(l, activation='relu')(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.Model(inputs, x)


def get_pretrain_model(n_hidden, days, N_OBSERVATION):
    obs_input = layers.Input((days, N_OBSERVATION))
    dS_input = layers.Input((days, 1))
    x = obs_input
    for l in n_hidden:
        x = layers.Dense(l, activation='relu')(x)
    x = layers.Dense(1, activation='tanh')(x)
    pretrain_actor = tf.keras.Model(obs_input, x)
    sum_hedge_pl = tf.reduce_sum(dS_input * x, axis=(1, 2))
    pretrainer = tf.keras.Model((obs_input, dS_input), sum_hedge_pl)
    return pretrain_actor, pretrainer
