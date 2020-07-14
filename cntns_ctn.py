import numpy as onp
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from torch.distributions.beta import Beta
from torch.optim.lr_scheduler import StepLR

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
    N_0 = S_0 + I_0 #total initial population size
    initial_conditions = onp.asarray((S_0, I_0, N_0))

    def dynamics(x, controls):
        S, I, N = x
        v, u, r = controls
        S_prime = - (beta / (1. + v)) * S * I * (1. / N) - u * S
        I_prime = (beta / (1. + v)) * S * I * (1. / N)  - alpha * I - r * I
        N_prime = - (1 - f) * alpha * I
        return onp.asarray((S_prime,
                            I_prime, 
                            N_prime))

    def cost(x, controls):
        S, I, N = x
        return (1 - f) * alpha * I + B * onp.linalg.norm(controls)

    return dynamics, initial_conditions, cost
  
class SIR(object):
    def __init__(self):
        super(SIR, self).__init__()
        
        self.alpha = 0.1
        self.beta = 0.5
        self.S_0 = 45000
        self.I_0 = 5000
        self.f, self.x, self.cost = make_SIR_Treatement_model(self.S_0, self.I_0, alpha=self.alpha, beta=self.beta, f=0.5, B=1)
        
    def step(self,action_vector):
        x_init = self.x
        solution = solve_ivp(lambda t,x : self.f(x, action_vector), [0,1], x_init ,t_eval=[1])
        x_next = onp.round(solution.y[:,-1])        
        self.x = x_next
        reward = -1 * self.cost(x_next, action_vector)
        done = False
        if self.x[1] < 1:
            done = True
        return self.x, reward, done, {}

    def reset(self):
        self.f, self.x, self.cost = make_SIR_Treatement_model(self.S_0, self.I_0, alpha=self.alpha, beta=self.beta, f=0.5, B=1)
        return self.x
    
    
def discount_rewards(rewards, gamma=0.99):
    r = onp.array([gamma**i * rewards[i]
    for i in range(len(rewards))])
    # Reverse the array direction for cumsum and then
    # revert back to the original order
    r = r[::-1].cumsum()[::-1]
    return r - r.mean()

def process(population_vector, N_0):
    x = population_vector/population_vector[2]
    x[2] = population_vector[2]/N_0
    return x

class policy_estimator(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.n_inputs = 3 #env.observation_space.shape[0]
        self.n_outputs = 1 #env.action_space.n
        # Define network
        self.body_out = nn.Sequential(
        nn.Linear(self.n_inputs, 128),
        nn.ReLU(),
        nn.Linear(128, 16))
        
        self.head_a = nn.Sequential(
        nn.Linear(16, 3))
        
        self.head_b = nn.Sequential(
        nn.Linear(16, 3))
        
        self.optimizer = optim.Adam(self.parameters(), lr= 0.0067)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.99)
        
        
    def foward(self, state):
        body_output = self.body_out(torch.FloatTensor(state))
        a = torch.exp(self.head_a(body_output))
        b = torch.exp(self.head_b(body_output))
        # a = self.head_a(body_output)
        # b = self.head_b(body_output)
        return a, b
    
    def update(self, a_tnsr, b_tnsr, action_tensor, reward_tensor):
        self.optimizer.zero_grad()
        m = Beta(a_tnsr, b_tnsr)   
        log_probs = m.log_prob(action_tensor)
        log_probs = -1* torch.matmul(reward_tensor, log_probs)   
        loss = log_probs.mean()   
        # print(loss)             
        loss.backward()
            
        self.optimizer.step()
        self.scheduler.step()       

def reinforce(env, policy_estimator, num_episodes=2000, batch_size=10, gamma=0.99):    
    total_rewards = []
    days_counter = []    
    batch_rewards = []
    batch_states = []
    batch_actions = []
    counter = 0       
    ep = 0
    days = 0
    
    while ep < num_episodes:
        # print(ep)
        s_0 = env.reset()
        days = 0
        states = []
        rewards = []
        actions = []
        done = False
        
        while done == False:     
            if days > 1000:
                print(days)
            
            processed_state = process(s_0, 50000)            
            a, b = policy_estimator.foward(processed_state)    
            distribution = Beta(a, b) 
            action = distribution.sample().detach().numpy() 
            s_1, r, done, _ = env.step(action)                       
            states.append(processed_state)
            rewards.append(r)
            actions.append(action)               
            days += 1
            counter += 1
            s_0 = s_1
               
            
        ep += 1 
        total_rewards.append(sum(rewards))            
        days_counter.append(days)
        
        if counter > 256 and done:  
#             print("reached")            
            returns = discount_rewards(rewards, gamma)              
            batch_states.extend(states)
            batch_rewards.extend(returns)
            batch_actions.extend(actions)          
            
            state_tensor = torch.FloatTensor(batch_states)
            reward_tensor = torch.FloatTensor(batch_rewards)
            a_tnsr, b_tnsr = policy_estimator.foward(state_tensor)
            action_tensor = torch.FloatTensor(batch_actions)
            policy_estimator.update(a_tnsr, b_tnsr, action_tensor, reward_tensor)
            
            batch_rewards = []
            batch_actions = []
            batch_states = []            
            counter = 0
#             print("finished")
    return total_rewards, days_counter

sum_returns = onp.zeros(2000)
sum_days = onp.zeros(2000)

for i in range(10):    
    env1 = SIR()
    policy_est1 = policy_estimator(env1)
    print(i)        
    returns1, days_counter1 = reinforce(env1, policy_est1)
    sum_returns += returns1
    sum_days += days_counter1

plt.figure(0)
plt.plot(range(2000), sum_returns/10)
plt.figure(1)
plt.plot(range(2000), sum_days/10)
