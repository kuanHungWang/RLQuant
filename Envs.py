import numpy as np
from numpy import exp

N_OBSERVATION = 5
from dataclasses import dataclass


class BlackProcess:
    def __init__(self, S0, r, sigma, n):
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.n = n

    def generate(self):
        S0, r, sigma, n = self.S0, self.r, self.sigma, self.n
        dt = 1 / 365
        dW = np.random.normal(0, dt ** 0.5, n)
        chg = np.ones(n + 1)
        chg[1:] += r * dt + sigma * dW
        accum_chg = chg.cumprod()
        return S0 * accum_chg


def create_black_process_close_form(n=10, S0=10, mu=0.01, sigma=0.3):
    np.random.seed(1)
    dt = 1 / 365
    dW = np.random.normal(0, dt ** 0.5, n)
    Wt = dW.cumsum()
    t = np.arange(1, 1 + n) / 365
    R_t = np.exp((mu - sigma ** 2 / 2) * t + sigma * Wt)
    return S0 * R_t


def MCCall(strike, n, m, r=0.01, sigma=0.9, S0=10):
    """

    :param strike: call strike
    :param n: option tenor in terms of  days
    :param m: number of simulation paths
    :return: option price by monte carlo
    """

    def create_St():
        dt = 1 / 365
        # dW=np.random.normal(0,dt**0.5,n)
        Wt = np.random.normal(0, (n / 365) ** 0.5)
        t = n / 365
        R_t = np.exp((r - sigma ** 2 / 2) * t + sigma * Wt)
        return S0 * R_t

    St = np.array([create_St() for _ in range(m)])

    payoff = np.clip(St - strike, 0, 999999)
    df = np.exp(-r * n / 365)
    return np.mean(payoff) * df


@dataclass
class Option:
    strike: float
    days: int
    isCall: bool = True


class VanillaEnv():
    n_observation = 5

    def __init__(self, process: BlackProcess, tenor, strike):
        self.process = process
        self.tenor = tenor
        self.strike = strike
        self.t = 0
        self.path = None
        self.observations = None
        self.reset()

    def df(self):
        """
        :return: discount factor for one day (gamma in terms of RL)
        """
        return exp(-self.process.r / 365)

    def mu(self):
        return exp(self.process.r / 365) - 1

    def reset(self):
        self.path = self.process.generate()
        self.t = 0
        self.observations = np.stack([self.observation(t) for t in range(self.tenor + 1)], 0)
        return self.observations[0]

    def St(self, t=None) -> np.float32:
        t = self.t if t is None else t
        return self.path[t]

    def observation(self, t=None):
        """
        :return: (x, x^2, y, y^2, xy)
        """
        S_K = self.St(t) / self.strike
        moneyness = max(0, S_K)

        t = self.t if t is None else t
        tenor = (self.tenor - t) / 365

        obs = np.array([moneyness, moneyness ** 2, tenor, tenor ** 2, moneyness * tenor])
        assert len(obs) == N_OBSERVATION
        return obs

    def step(self, action):
        """

        :param action:
        :return: S_t0, S_t1, reward, terminated, can_early_exercise, payoff, dS
        """
        S_t0 = self.observations[self.t]
        self.t = self.t + 1
        dS = self.St() - self.St(self.t - 1)
        reward = dS * action
        S_t1 = self.observations[self.t]
        terminated = True if self.t >= self.tenor else False
        can_early_exercise = False
        payoff = self.payoff()
        return S_t0, S_t1, reward, terminated, can_early_exercise, payoff, dS

    def payoff(self, t=None) -> np.float32:
        """
        :return: option payoff if exercise now, regardless it can be exercised, equivalent to moneyless
        """
        return max(0, self.St(t) - self.strike)
