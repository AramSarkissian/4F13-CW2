# Metropolis-Hastings MCMC algorithm for sampling skills in the probit rank model
# -gtc 20/09/2025
import numpy as np
from scipy.stats import norm
from tqdm import tqdm

def MH_sample(games, num_players, num_its):

    # pre-process data:
    # array of games for each player, X[i] = [(other_player, outcome), ...]
    X = [[] for _ in range(num_players)] 
    for a, (i,j) in enumerate(games):
        X[i].append((j, +1))  # player i beat player j
        X[j].append((i, -1))  # player j lost to payer i
    for i in range(num_players):
        X[i] = np.array(X[i])

    # array that will contain skill samples
    skill_samples = np.zeros((num_players, num_its))

    w = np.zeros(num_players)  # skill for each player

    sigma_prop = 0.1  # proposal std dev
    w_mean = []
    w_std = []

    for itr in tqdm(range(num_its)):
        for i in range(num_players):
            j, outcome = X[i].T

            # current local log-prob 
            lp1 = norm.logpdf(w[i]) + np.sum(norm.logcdf(outcome*(w[i]-w[j])))

            # proposed new skill and log-prob
            w_prop = w[i] + np.random.normal(scale=sigma_prop)
            lp2 = norm.logpdf(w_prop) + np.sum(norm.logcdf(outcome*(w_prop - w[j])))
            acc_prob = np.exp(lp2 - lp1)
            
            # accept or reject move:
            if np.random.rand() < acc_prob:
                w[i] = w_prop

        skill_samples[:, itr] = w
        w_mean.append(np.mean(w))
        w_std.append(np.std(w))


    return skill_samples, w_mean, w_std


