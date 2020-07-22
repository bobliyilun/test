#ipython code
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#ipython format, so don't run this .py file directly
!pip uninstall -y pyarrow
!pip install ray[debug]==0.7.5
!pip install bs4
!pip install lz4

import numpy as onp
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.distributions.beta import Beta

import gym
from gym import spaces
import numpy as np
from tutorial.rllib_exercises import test_exercises

import ray
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG

ray.init(ignore_reinit_error=True, log_to_driver=False)


def make_SIR_Treatement_model(
        S_0,
        I_0,
        alpha=2,
        beta=2,
        f=1,
        B=1
):
    '''
    beta: transmission coefficient
    alpha: rate of infectives leaving the infectious class
    f: proportion of infectives recovering and going to the removed class, with
        the remainder dying of infection
    B: cost of applying control u
    '''
    N_0 = S_0 + I_0  # total initial population size
    initial_conditions = onp.asarray((S_0, I_0, N_0))

    def dynamics(x, controls):
        S, I, N = x
        v, u, r = controls
        S_prime = - (beta / (1. + v)) * S * I * (1. / N) - u * S
        I_prime = (beta / (1. + v)) * S * I * (1. / N) - alpha * I - r * I
        N_prime = - (1 - f) * alpha * I
        return onp.asarray((S_prime,
                            I_prime,
                            N_prime))

    def cost(x, controls):
        S, I, N = x
        return (1 - f) * alpha * I + B * onp.linalg.norm(controls)

    return dynamics, initial_conditions, cost


class SIR(gym.Env):
    def __init__(self, env_config=None):
        super(SIR, self).__init__()

        self.action_space = spaces.Box(np.array([0, 0, 0]), np.array([1, 1, 1]))
        self.observation_space = spaces.MultiDiscrete([5001, 5001, 5001])

        self.alpha = 0.1
        self.beta = 0.5
        self.S_0 = 4500
        self.I_0 = 500
        self.f, self.x, self.cost = make_SIR_Treatement_model(self.S_0, self.I_0, alpha=self.alpha, beta=self.beta, f=0,
                                                              B=1)

    def step(self, action_vector):
        x_init = self.x
        solution = solve_ivp(lambda t, x: self.f(x, action_vector), [0, 1], x_init, t_eval=[1])
        x_next = onp.round(solution.y[:, -1])
        self.x = x_next
        reward = -1 * self.cost(x_next, action_vector)
        done = False
        if self.x[1] < 1:
            done = True
        return self.x, reward, done, {}

    def reset(self):
        self.f, self.x, self.cost = make_SIR_Treatement_model(self.S_0, self.I_0, alpha=self.alpha, beta=self.beta,
                                                              f=0.5, B=1)
        return self.x




trainer_config = DEFAULT_CONFIG.copy()
trainer_config['num_workers'] = 1
trainer_config["train_batch_size"] = 400
trainer_config["sgd_minibatch_size"] = 64
trainer_config["num_sgd_iter"] = 10




trainer = PPOTrainer(trainer_config, SIR);
for i in range(200):
    print("Training iteration {}...".format(i))
    trainer.train()



env = SIR()
state = env.reset()

done = False
#max_state = -1
cumulative_reward = 0

total_states = list()
while not done:
    action = trainer.compute_action(state)
    state, reward, done, results = env.step(action)
    #max_state = max(max_state, state)
    total_states.append(state)
    cumulative_reward += reward

print("Cumulative reward you've received is: {}. Congratulations!".format(cumulative_reward))
print("Final state is", state)