import numpy as np
import tensorflow as tf

from Envs import VanillaEnv, BlackProcess
from model import get_actor, get_critic, get_pretrain_model
from replay_buffer import EpisodeBuffer


def single_sample(dataset, i):
    observations, dS, payoff = dataset
    return observations[i:i + 1, :, :], dS[i:i + 1, :, :], payoff[i:i + 1]


def gather_episode_wise(env, buffer: EpisodeBuffer, episodes, action=0.5):
    for i in range(episodes):
        env.reset()
        while True:
            data = env.step(action)  # actual delta still doesn't matter, avoid calling actor to save time
            done = data[3]
            buffer.store(data, env.t - 1)
            if done:
                break


def gather_data_pretrain_critic(pretrain_actor, env, n_samples):
    df = env.df()
    days = env.tenor
    buffer = EpisodeBuffer(n_samples, days, VanillaEnv.n_observation)
    gather_episode_wise(env, buffer, n_samples)
    S_t0, S_t1, reward, terminated, can_early_exercise, payoff, dS = buffer.storage
    y_t1 = payoff[:, -1, :]
    Y_hedge = np.zeros_like(payoff, dtype=np.float32)
    for i in reversed(range(days)):
        def reshape(arr):
            return arr[:, i, :]

        hedge_pl = reshape(dS) * pretrain_actor(reshape(S_t0)).numpy()
        y_t0 = y_t1 * df - hedge_pl
        y_t0 = np.maximum(0, y_t0)
        Y_hedge[:, i, :] = y_t0
        y_t1 = y_t0
    return Y_hedge, buffer


# build pretrain model of actor
"""
def get_pretrain_model(n_hidden):
    obs_input = layers.Input((days, VanillaEnv.n_observation))
    dS_input = layers.Input((days, 1))
    x=obs_input
    for l in n_hidden:
        x = layers.Dense(l, activation='relu')(x)
    x = layers.Dense(1, activation='tanh')(x)
    pretrain_actor = tf.keras.Model(obs_input, x)
    sum_hedge_pl = tf.reduce_sum(dS_input * x, axis=(1, 2))
    pretrainer = tf.keras.Model((obs_input, dS_input), sum_hedge_pl)
    return pretrain_actor, pretrainer
"""


def get_pretrain_actor(env, n_hidden, n_samples, epoch=12):
    N_OBSERVATION = env.n_observation
    days = env.tenor
    buffer = EpisodeBuffer(n_samples, days, VanillaEnv.n_observation)
    gather_episode_wise(env, buffer, n_samples)
    observations, dS, payoff = buffer.storage[0], buffer.storage[-1], buffer.storage[-2][:, -1, 0]
    pretrain_actor, pretrainer = get_pretrain_model(n_hidden, days, N_OBSERVATION)  #
    pretrainer.compile(loss=tf.keras.losses.mse, optimizer="Adam")
    pretrainer.fit((observations, dS), payoff, 64, epoch)
    actor = get_actor(n_hidden, N_OBSERVATION)
    actor.set_weights(pretrain_actor.get_weights())
    return actor


def get_pretrain_critic(env, actor, n_hidden, N_OBSERVATION, n_samples, epoc=10):
    Y_hedge, buffer = gather_data_pretrain_critic(actor, env, n_samples)
    S_t0, S_t1, reward, terminated, can_early_exercise, payoff, dS = buffer.storage

    def reshape(arr):
        return arr.reshape((-1, arr.shape[-1]))

    critic = get_critic(n_hidden, N_OBSERVATION)
    critic.compile(loss=tf.keras.losses.mse, optimizer="Adam")
    critic.fit(reshape(S_t0), reshape(Y_hedge), 64, epoc)
    return critic, buffer


# use short term, OTM option
if __name__ == '__main__':
    S0, r, vol, days, strike = 1, 0.01, 0.3, 30, 1.1
    process = BlackProcess(S0, r, vol, days)
    N_OBSERVATION = VanillaEnv.n_observation
    env = VanillaEnv(process, days, strike)
    n_samples = 2 ** 12
    n_hidden = [64, 64]
    actor = get_pretrain_actor(env, n_hidden, n_samples, epoch=10)
    # create a new env for ATM option
    env = VanillaEnv(process, days, S0)

    critic, buffer = get_pretrain_critic(env, actor, n_hidden, N_OBSERVATION, epoc=5)
    S_t0, S_t1, reward, terminated, can_early_exercise, payoff, dS = buffer.storage

    print(critic(S_t0[:, 0]).numpy().transpose())
    print(critic(S_t0[:, days - 1]).numpy().transpose())
    print(payoff[:, days - 1].transpose())
