import numpy as np
import random

class NonStationaryBandit:
    def __init__(self, num_arms=10, initial_mean=0.0):
        self.num_arms = num_arms
        self.q_star = np.full(num_arms, initial_mean, dtype=float)
        
    def step(self, action):
        self.q_star += np.random.normal(loc=0.0, scale=0.01, size=self.num_arms)
        
        true_mean = self.q_star[action - 1]
        value = np.random.normal(loc=true_mean, scale=1.0)
        
        optimal_action = np.argmax(self.q_star) + 1
        
        return value, optimal_action

if __name__ == '__main__':
    bandit = NonStationaryBandit()
    reward, opt_action = bandit.step(1)
    print(f"Part 3 Environment Test: Action 1 returned reward {reward:.4f}. Current Optimal Action: {opt_action}")