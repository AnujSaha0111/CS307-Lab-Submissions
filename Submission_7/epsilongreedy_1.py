import random

def binary_bandit(action, bandit_case='A'):
    action_index = action - 1 
    
    if bandit_case == 'A':
        p = [0.1, 0.2] 
    elif bandit_case == 'B':
        p = [0.8, 0.9]
    else:
        raise ValueError("Invalid bandit_case")

    if random.random() < p[action_index]:
        value = 1
    else:
        value = 0
    return value

def run_epsilon_greedy(num_actions, epsilon, total_steps, bandit_case):
    Q = [0.0] * num_actions  
    N = [0] * num_actions  
    
    rewards_history = []
    
    for t in range(1, total_steps + 1):
        if random.random() < epsilon:
            action = random.randint(1, num_actions)
        else:
            max_q = max(Q)
            best_actions = [a + 1 for a, q in enumerate(Q) if q == max_q]
            action = random.choice(best_actions)
            
        reward = binary_bandit(action, bandit_case)
        
        action_index = action - 1
        N[action_index] += 1
        
        alpha = 1.0 / N[action_index]
        
        Q[action_index] += alpha * (reward - Q[action_index])
        
        rewards_history.append(reward)

    avg_reward = sum(rewards_history) / total_steps
    
    print(f"\nResults for Binary Bandit {bandit_case} (Steps={total_steps}, epsilon={epsilon}) ")
    print(f"Final Action-Value Estimates (Q): {Q}")
    print(f"Total Action Counts (N): {N}")
    print(f"Average Reward over {total_steps} steps: {avg_reward:.4f}")
    
    true_means = [0.1, 0.2] if bandit_case == 'A' else [0.8, 0.9]
    optimal_action = 2 if true_means[1] > true_means[0] else 1
    
    if max(Q) == Q[optimal_action - 1]:
        print(f"Agent successfully latched onto the optimal action: {optimal_action}")
    else:
        print(f"Agent may have converged to a sub-optimal action.")

TOTAL_STEPS = 5000 
EPSILON = 0.1 

print("\n" + "="*50)
print("Part 2: Epsilon-Greedy on Stationary Binary Bandits")
print("="*50)
run_epsilon_greedy(num_actions=2, epsilon=EPSILON, total_steps=TOTAL_STEPS, bandit_case='A')
run_epsilon_greedy(num_actions=2, epsilon=EPSILON, total_steps=TOTAL_STEPS, bandit_case='B')