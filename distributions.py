import random
from functools import reduce

import numpy as np

def _check_unity(weights, tol=1e-8):
    return abs(np.sum(weights) - 1) < tol

class discrete_multinomial:
    def __init__(self, choices=None, weights=None):

        if weights is None and choices is None:
            raise ValueError("Set at least one of weights or choices")

        if choices is None:
            choices = list(range(len(weights)))
        if weights is None:
            weights = [1. / len(choices)] * len(choices)

        if not _check_unity(weights):
            raise ValueError("Weights do not add up to 1")

        self.weights = weights
        self.choices = choices

    def sample(self, n=None):
        x = random.choices(self.choices, self.weights, k=(n or 1))
        return x if n else x[0]

    def log_density(self, x):
        try:
            i = self.choices.index(x)
            return np.log(self.weights[i])
        except ValueError:
            return -np.inf

class continuous_multivariate_gaussian:
    def __init__(self, mu, sigma):

        if isinstance(mu, float):
            mu = [mu]

        if isinstance(sigma, float):
            self.sigma = np.eye(len(mu))
        # 1d convariance should be used as diagonal:
        elif isinstance(sigma[0], float):
            self.sigma = np.diag(sigma)
        else:
            self.sigma = sigma

        self.mu = np.array(mu)

    def sample(self, n=None):
        x = np.random.multivariate_normal(self.mu, self.sigma, size=((n or 1),))
        return x if n else x[0]

    def log_density(self, x):
        d = len(self.mu)
        z = np.array(x) - np.array(self.mu)
        zz = (z @ np.linalg.inv(self.sigma) @ z)
        return -0.5 * (zz + d * np.log(2 * np.pi) + np.log(np.linalg.det(self.sigma)))


class continuous_multivariate_uniform:
    def __init__(self, x0, delta_x):

        if isinstance(x0, float):
            x0 = [x0]

        if isinstance(delta_x, float):
            self.dx = np.array([delta_x] * len(x0))
        else:
            self.dx = np.array(delta_x)

        self.x0 = np.array(x0)

    def sample(self, n=None):
        u = np.random.random(size=((n or 1),len(self.x0)))
        x = u * self.dx[np.newaxis, :] + self.x0[np.newaxis, :]
        return x if n else x[0]

    def log_density(self, x):
        u = x - self.x0
        u *= np.sign(self.dx)
        in_interval = np.all(u >= 0) and np.all(u < np.abs(self.dx))

        if in_interval:
            return -np.sum(np.log(np.abs(self.dx)))
        return -np.inf


class continuous_mixture:

    def __init__(self, distributions, weights=None):

        if weights is None:
            weights = [1. / len(distributions)] * len(distributions)

        if not _check_unity(weights):
            raise ValueError("Weights do not add up to 1")

        self.weights = weights
        self.distributions = distributions

    def sample(self, n=None):

        n_per_distrib = np.random.multinomial((n or 1), self.weights)
        x_per_distrib = []

        for i, n_i in enumerate(n_per_distrib):
            if n_i == 0:
                continue
            x_per_distrib.append(self.distributions[i].sample(n_i))

        x = np.concatenate(x_per_distrib)
        np.random.shuffle(x)

        return x if n else x[0]

    def log_density(self, x):
        log_pj = [np.log(w) + d.log_density(x) for w,d in zip(self.weights, self.distributions)]
        return reduce(np.logaddexp, log_pj)


if __name__=='__main__':


    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    decaying_weights = np.array([1. / k**2 for k in range(1,8)])
    decaying_weights /= np.sum(decaying_weights)

    gauss    = continuous_multivariate_gaussian
    uniform  = continuous_multivariate_uniform
    discrete = discrete_multinomial
    mix      = continuous_mixture

    distribs = [
                'discrete(weights=decaying_weights)',
                'discrete(choices=[1.1, 1.5, 4.6, 5.9])',
                'gauss(5.0, 3.0)',
                'uniform([6], [2])',
                'mix([gauss(7., 2.), uniform(2.5, 3.)], weights=[0.7, 0.3])',
                'mix([uniform(5., 3.), uniform(5., -1.5)])'
            ]

    plot_w = 3
    plot_h = len(distribs) // plot_w + (1 if (len(distribs) % plot_w) else 0)

    plot_range = (0, 10)
    plot_axis = np.linspace(*plot_range, num=1000, endpoint=False)

    plt.figure(figsize=(4*plot_w, 3*plot_h))

    for i, d in enumerate(distribs):
        plt.subplot(plot_h, plot_w, i+1)

        dist = eval(d)
        samples = dist.sample(n=64_000)
        likelihoods = np.array([dist.log_density(x) for x in plot_axis])
        likelihoods = np.exp(likelihoods)

        plt.plot(plot_axis, likelihoods, '--', color='crimson')
        plt.hist(samples, bins=100, range=plot_range, density=True,
                 histtype='step', alpha=0.9, color='royalblue')
        plt.xlim(*plot_range)

        plt.gca().set_title(d)

    plt.tight_layout()
    plt.savefig('ProbabilityDistribExamples.pdf')
