import numpy as np


class EpisodeBuffer():
    def __init__(self, capacity, ep_length, N_OBSERVATION):
        self.capacity = capacity
        self.ep_count = 0

        def blank_array(dim):
            assert dim <= 2
            dim = N_OBSERVATION if dim == 2 else 1
            return np.zeros((capacity, ep_length, dim), dtype=np.float32)

        self.storage = [blank_array(2), blank_array(2),
                        blank_array(1), blank_array(1), blank_array(1), blank_array(1), blank_array(1)]
        # order: S_t0, S_t1, reward, terminated, can_early_exercise, payoff, dS

    def store(self, values, t):
        ep_index = self.ep_count % self.capacity
        for storage, value in zip(self.storage, values):
            storage[ep_index, t, :] = value
        done = values[3]
        if done:
            self.ep_count = self.ep_count + 1

    def sample(self, batch_size):
        indexes = np.random.choice(self.capacity, batch_size, False)
        return [v[indexes] for v in self.storage]


class Buffer():
    """
    S_t0, S_t1, reward, terminated, can_early_exercise, payoff, dS
    """

    def __init__(self, size, N_OBSERVATION):
        self.size = size

        def blank_array(dim):
            assert dim <= 2
            dim = N_OBSERVATION if dim == 2 else 1
            return np.zeros((size, dim), dtype=np.float32)

        self.storage = [blank_array(2), blank_array(2),
                        blank_array(1), blank_array(1), blank_array(1), blank_array(1), blank_array(1)]
        # order: S_t0, S_t1, reward, terminated, can_early_exercise, payoff, dS
        self.count = 0

    def store(self, values):
        index = self.count % self.size
        for storage, value in zip(self.storage, values):
            storage[index, :] = value
        self.count = self.count + 1

    def sample(self, batch_size):
        indexes = np.random.choice(self.size, batch_size, False)
        return [v[indexes] for v in self.storage]
