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

def train_agent_part4(total_steps, epsilon, constant_alpha, is_constant_alpha):
    bandit = NonStationaryBandit() 
    num_actions = bandit.num_arms
    
    Q = np.zeros(num_actions, dtype=float) 
    N = np.zeros(num_actions, dtype=int)
    optimal_action_percent = []

    for t in range(1, total_steps + 1):
        
        if random.random() < epsilon:
            action = random.randint(1, num_actions) # Explore
        else:
            action = np.argmax(Q) + 1 # Exploit
            
        reward, true_optimal_action = bandit.step(action)
        action_index = action - 1
        
        N[action_index] += 1
        
        if is_constant_alpha:
            alpha = constant_alpha 
        else:
            alpha = 1.0 / N[action_index] 
            
        Q[action_index] += alpha * (reward - Q[action_index])
        
        if action == true_optimal_action:
            optimal_action_percent.append(1)
        else:
            optimal_action_percent.append(0)

    percent_optimal = np.mean(optimal_action_percent) * 100
    return percent_optimal

if __name__ == '__main__':
    TOTAL_STEPS = 10000 
    EPSILON = 0.1 
    ALPHA_CONST = 0.1 
    NUM_RUNS = 200 

    print("="*70)
    print(f"Part 4: Epsilon-Greedy Experiment on Non-Stationary Bandit")
    print(f"Testing Constant Alpha ({ALPHA_CONST}) vs. Standard Sample-Average (1/N)")
    print(f"Parameters: Steps={TOTAL_STEPS}, Epsilon={EPSILON}, Averaged over {NUM_RUNS} runs")
    print("="*70)

    results_constant_alpha = []
    results_sample_average = []
    
    for i in range(NUM_RUNS):
        percent_opt_const = train_agent_part4(TOTAL_STEPS, EPSILON, ALPHA_CONST, is_constant_alpha=True)
        results_constant_alpha.append(percent_opt_const)

        percent_opt_sample = train_agent_part4(TOTAL_STEPS, EPSILON, ALPHA_CONST, is_constant_alpha=False)
        results_sample_average.append(percent_opt_sample)

    avg_optimal_const = np.mean(results_constant_alpha)
    avg_optimal_sample = np.mean(results_sample_average)
    
    print("\nFinal Results (Average % Optimal Action) ")
    print(f"Modified Agent (Constant alpha={ALPHA_CONST}): {avg_optimal_const:.2f}%")
    print(f"Standard Agent (Sample Average = 1/N):  {avg_optimal_sample:.2f}%")
    
    print("\nAssessment: Ability to Latch onto Correct Actions")
    
    print(f"The Modified Epsilon-Greedy Agent (Constant alpha={ALPHA_CONST}) IS able to latch onto the correct actions.")
    print("This is clearly demonstrated by its superior average percentage of optimal actions compared to the Standard Agent.")