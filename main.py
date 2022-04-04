import numpy as np
import tensorflow as tf

from Envs import BlackProcess, VanillaEnv
from blackscholes import call, delta
from model import get_critic
from pretrain import get_pretrain_actor, get_pretrain_critic
from replay_buffer import Buffer

N_OBSERVATION = VanillaEnv.n_observation
S0, mu, sigma = 100, 0.5, 0.8

"""def hedge_payoff_error(delta, dS, payoff):

    return tf.reduce_sum(delta * dS)
"""


def next_valuation(terminated, model_valuation, payoff):
    return terminated * payoff + (1 - terminated) * model_valuation


def get_critic_loss(model, target_model, S_t0, S_t1, reward, terminated, payoff, df):
    learn_target = next_valuation(terminated, target_model(S_t1), payoff) * df - reward
    return tf.math.reduce_mean(tf.math.square(learn_target - model(S_t0)))


def get_actor_loss(actor, critic, S_t0, S_t1, dS, df, mu):
    delta = actor(S_t0)
    hedge_pl = delta * (dS - mu)
    learn_target = critic(S_t1) * df - critic(S_t0)
    return tf.math.reduce_mean(tf.math.square(learn_target - hedge_pl))


def should_exercise(model, s, payoff) -> bool:
    valuation = model(s)
    return valuation > payoff


@tf.function
def learn(train_data, actor, critic, critic_target, optimizer_critic, optimizer_actor, df, mu):
    S_t0, S_t1, reward, terminated, can_early_exercise, payoff, dS = train_data
    with tf.GradientTape() as tape:
        # todo: dcf shold apply interest rate
        critic_loss = get_critic_loss(critic, critic_target, S_t0, S_t1, reward, terminated, payoff, df)
        critic_gradient = tape.gradient(critic_loss, critic.trainable_variables)
        optimizer_critic.apply_gradients(zip(critic_gradient, critic.trainable_variables))
    with tf.GradientTape() as tape:
        actor_loss = get_actor_loss(actor, critic, S_t0, S_t1, dS, df, mu)
        actor_gradient = tape.gradient(actor_loss, actor.trainable_variables)
        optimizer_actor.apply_gradients(zip(actor_gradient, actor.trainable_variables))


@tf.function
def soft_update(target_weights, weights, tau):
    for (old_value, new_value) in zip(target_weights, weights):
        old_value.assign(new_value * tau + old_value * (1 - tau))


if __name__ == '__main__':
    S0, r, vol, days, strike = 1, 0.01, 0.3, 30, 1.1
    n_samples = 2 ** 12
    n_hidden = [64, 64]
    process = BlackProcess(S0, r, vol, days)
    N_OBSERVATION = VanillaEnv.n_observation
    env = VanillaEnv(process, days, strike)
    print("pretrain actor")
    actor = get_pretrain_actor(env, n_hidden, n_samples)
    env = VanillaEnv(process, days, S0)
    print("pretrain critic")
    critic, _ = get_pretrain_critic(env, actor, n_hidden, N_OBSERVATION, n_samples, epoc=5)
    critic_target = get_critic(n_hidden, N_OBSERVATION)
    critic_target.set_weights(critic.get_weights())
    optimizer_critic = tf.keras.optimizers.Adam(0.002)
    optimizer_actor = tf.keras.optimizers.Adam(0.002)
    learn_per_step = 1
    buffer = Buffer(1024, N_OBSERVATION)
    batch_size = 32
    tau = 0.1
    df = env.df()
    mu = env.mu()
    print("train like actor-critic")
    for eps in range(50):
        print("episode {}".format(eps))
        epsode_reward = 0
        S_t0 = env.reset()[np.newaxis, :]
        epsode_reward += critic(S_t0)
        while True:
            action = actor(S_t0)
            step_result = env.step(action)
            _, S_t1, reward, terminated, can_early_exercise, payoff, dS = step_result
            epsode_reward += reward

            buffer.store(step_result)
            if buffer.count > batch_size:
                train_data = buffer.sample(batch_size)
                for _ in range(learn_per_step):
                    learn(train_data, actor, critic, critic_target, optimizer_critic, optimizer_actor, df, mu)
                soft_update(critic_target.variables, critic.variables, tau)
            if terminated:
                print("total hedge P/L: {:5.4f}, option payoff: {:5.4f}".format(epsode_reward[0, 0], payoff))
                break
            S_t0 = S_t1[np.newaxis, :]
    S_t0 = env.reset()[np.newaxis, :]
    print("by RL model: \n option value: {:5.4f}, delta: {:5.4f}".format(critic(S_t0)[0, 0], actor(S_t0)[0, 0]))
    bs_call = call(S0, S0, days / 365, r, 0, vol)
    bs_delta = delta(S0, S0, days / 365, r, 0, vol, True)
    print("by black-scholes model: \n option value: {:5.4f}, delta: {:5.4f}".format(bs_call, bs_delta))
