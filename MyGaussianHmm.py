import numpy as np
import scipy.stats as st


class MyGaussianHmm:
    def __init__(self, N_states, n_dims=None):
        # self.emission_probability = np.zeros((N_states, M_obs))
        self.s = np.random.RandomState(0)
        self.N_states = N_states
        self.initial_states = self.s.rand(self.N_states)

        self.initial_states /= self.initial_states.sum()
        self.transition_probability = self.s.rand(self.N_states, self.N_states)
        self.transition_probability /= self.transition_probability.sum(axis=1)
        # self.M_obs = M_obs
        self.n_dims = n_dims

    def forward(self, emission_probability):
        alpha = np.zeros(emission_probability.shape)
        alpha[:, 0] = self.initial_states * emission_probability[:, 0]

        alpha[:, 0] = self.normalize(alpha[:, 0])
        log_likely = alpha[:, 0].sum()
        log_likely = np.log(log_likely)

        # print(alpha)
        for t in range(1, emission_probability.shape[1]):
            alpha[:, t] = emission_probability[:, t] * (self.transition_probability.T @ alpha[:, t - 1])

            alpha_sum = alpha[:, t].sum()
            alpha[:, t] = self.normalize(alpha[:, t])  # alpha_sum
            log_likely += np.log(alpha_sum)
        return log_likely, alpha, alpha[:, -1].sum()

    def backward(self, emission_probability):
        beta = np.zeros(emission_probability.shape)
        beta[:, -1] = 1.
        for t in range(emission_probability.shape[1] - 2, -1, -1):
            beta[:, t] = self.transition_probability @ (emission_probability[:, t + 1] * beta[:, t + 1])
            beta[:, t] = self.normalize(beta[:, t])

        return beta

    def emission_distribution(self, obs):
        emission = np.zeros((self.N_states, obs.shape[1]))
        for s in range(self.N_states):
            emission[s, :] = st.multivariate_normal.pdf(obs.T, mean=self.mu[:, s].T, cov=self.covs[:, :, s].T)
        return emission / emission.sum()

    def init(self, obs):
        self.n_dims = obs.shape[0]
        subset = self.s.choice(np.arange(self.n_dims), size=self.N_states, replace=False)
        self.mu = obs[:, subset]
        self.covs = np.zeros((self.n_dims, self.n_dims, self.N_states))
        self.covs += np.diag(np.diag(np.cov(obs)))[:, :, None]

    def normalize(self, x):
        return (x + (x == 0)) / np.sum(x)

    def norm_axis1(self, x):
        return (x + (x == 0)) / np.sum(x, axis=1)

    def algorithm(self, obs):
        emission = self.emission_distribution(obs)
        self.emission = emission
        log_likely, alpha, _ = self.forward(emission)
        beta = self.backward(emission)
        self.alpha = alpha
        self.beta = beta
        gamma = np.zeros_like(alpha)
        psi = np.zeros((self.N_states, self.N_states, alpha.shape[1]))
        for t in range(alpha.shape[1]):
            alpha_and_beta = alpha[:, t] * beta[:, t]
            alpha_beta_sum = alpha_and_beta.sum()
            gamma[:, t] = alpha_and_beta / alpha_beta_sum
            if t == alpha.shape[1] - 1:
                break

            for s in range(self.N_states):
                psi[s, :, t] = alpha[s, t] * self.transition_probability[s, :] * emission[:, t + 1] * beta[:, t + 1]
            psi[:, :, t] = self.normalize(psi[:, :, t])
        self.psi = psi

        gamma_sum = gamma.sum(axis=1)
        psi_sum = psi.sum(axis=2)
        new_transition_probs = psi_sum / psi_sum.sum(axis=0)
        new_initial_probs = gamma[:, 0]
        new_mean = np.zeros((self.n_dims, self.N_states))
        new_covs = np.zeros((self.n_dims, self.n_dims, self.N_states))
        for s in range(self.N_states):
            gamma_obs = obs * gamma[s, :]
            new_mean[:, s] = np.sum(gamma_obs, axis=1) / gamma_sum[s]
            for n in range(obs.shape[1]):
                obs_minus_mean = (obs[:, n] - new_mean[:, s]).flatten()
                new_covs[:, :, s] += gamma[s, n] * np.multiply.outer(obs_minus_mean, obs_minus_mean)
            new_covs[:, :, s] /= gamma_sum[s]

        new_covs += .01 * np.eye(self.n_dims)[:, :, None]
        self.mu = new_mean
        self.transition_probability = new_transition_probs
        self.initial_states = new_initial_probs
        # self.covs = new_covs
        self.gamma = gamma
        return log_likely

    def score(self, obs):
        em = self.emission_distribution(obs)
        log_likelihood, _, _ = self.forward(em)
        return log_likelihood

    def train(self, X, n_iter=10):
        self.init(X)
        for i in range(n_iter):

            loglik = self.algorithm(X)
            # if i % 100 == 0:
            #     print(loglik)
