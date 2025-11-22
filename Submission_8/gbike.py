import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt

MAX_BIKES = 20
MAX_MOVE = 5
RENTAL_REWARD = 10
MOVE_COST = 2
DISCOUNT = 0.9

RENTAL_REQUEST_LOC1 = 3
RENTAL_REQUEST_LOC2 = 4
RETURN_LOC1 = 3
RETURN_LOC2 = 2

POISSON_UPPER = 11

PARKING_THRESHOLD = 10
EXTRA_PARKING_COST = 4

poisson_rental_1 = [poisson.pmf(i, RENTAL_REQUEST_LOC1) for i in range(POISSON_UPPER)]
poisson_rental_2 = [poisson.pmf(i, RENTAL_REQUEST_LOC2) for i in range(POISSON_UPPER)]
poisson_return_1 = [poisson.pmf(i, RETURN_LOC1) for i in range(POISSON_UPPER)]
poisson_return_2 = [poisson.pmf(i, RETURN_LOC2) for i in range(POISSON_UPPER)]

def build_dynamics(p_req, p_ret):
    trans = np.zeros((MAX_BIKES + 1, MAX_BIKES + 1))
    exp_reward = np.zeros(MAX_BIKES + 1)
    for s in range(MAX_BIKES + 1):
        for req in range(POISSON_UPPER):
            for ret in range(POISSON_UPPER):
                prob = p_req[req] * p_ret[ret]
                rentals = min(s, req)
                s_next = min(s - rentals + ret, MAX_BIKES)
                trans[s, s_next] += prob
                exp_reward[s] += prob * rentals * RENTAL_REWARD
    return trans, exp_reward

TRANS_1, EXP_REWARD_1 = build_dynamics(poisson_rental_1, poisson_return_1)
TRANS_2, EXP_REWARD_2 = build_dynamics(poisson_rental_2, poisson_return_2)

def expected_return_basic(s1, s2, action, V):
    returns = -MOVE_COST * abs(action)
    s1_morning = int(min(max(s1 - action, 0), MAX_BIKES))
    s2_morning = int(min(max(s2 + action, 0), MAX_BIKES))
    returns += EXP_REWARD_1[s1_morning] + EXP_REWARD_2[s2_morning]
    for s1_next in range(MAX_BIKES + 1):
        for s2_next in range(MAX_BIKES + 1):
            prob = TRANS_1[s1_morning, s1_next] * TRANS_2[s2_morning, s2_next]
            if prob > 0:
                returns += DISCOUNT * prob * V[s1_next, s2_next]
    return returns

def expected_return_modified(s1, s2, action, V):
    if action > 0:
        returns = -MOVE_COST * (action - 1)
    else:
        returns = -MOVE_COST * abs(action)
    s1_morning = int(min(max(s1 - action, 0), MAX_BIKES))
    s2_morning = int(min(max(s2 + action, 0), MAX_BIKES))
    if s1_morning > PARKING_THRESHOLD:
        returns -= EXTRA_PARKING_COST
    if s2_morning > PARKING_THRESHOLD:
        returns -= EXTRA_PARKING_COST
    returns += EXP_REWARD_1[s1_morning] + EXP_REWARD_2[s2_morning]
    for s1_next in range(MAX_BIKES + 1):
        for s2_next in range(MAX_BIKES + 1):
            prob = TRANS_1[s1_morning, s1_next] * TRANS_2[s2_morning, s2_next]
            if prob > 0:
                returns += DISCOUNT * prob * V[s1_next, s2_next]
    return returns

def policy_evaluation(policy, V, exp_return_func, theta=0.1):
    while True:
        delta = 0
        for i in range(MAX_BIKES + 1):
            for j in range(MAX_BIKES + 1):
                v = V[i, j]
                V[i, j] = exp_return_func(i, j, policy[i, j], V)
                delta = max(delta, abs(v - V[i, j]))
        if delta < theta:
            break
    return V

def policy_improvement(policy, V, exp_return_func):
    policy_stable = True
    for i in range(MAX_BIKES + 1):
        for j in range(MAX_BIKES + 1):
            old_action = policy[i, j]
            best_value = float('-inf')
            best_action = 0
            min_action = -min(j, MAX_BIKES - i, MAX_MOVE)
            max_action = min(i, MAX_BIKES - j, MAX_MOVE)
            for action in range(min_action, max_action + 1):
                value = exp_return_func(i, j, action, V)
                if value > best_value:
                    best_value = value
                    best_action = action
            policy[i, j] = best_action
            if old_action != best_action:
                policy_stable = False
    return policy, policy_stable

def policy_iteration(exp_return_func, name):
    print("\n" + "="*40)
    print(name)
    print("="*40)
    V = np.zeros((MAX_BIKES + 1, MAX_BIKES + 1))
    policy = np.zeros((MAX_BIKES + 1, MAX_BIKES + 1), dtype=int)
    while True:
        V = policy_evaluation(policy, V, exp_return_func)
        policy, stable = policy_improvement(policy, V, exp_return_func)
        if stable:
            break
    return V, policy

def plot_policy(policy, title):
    plt.figure(figsize=(6,5))
    plt.imshow(policy, origin='lower', cmap='bwr', vmin=-MAX_MOVE, vmax=MAX_MOVE)
    plt.colorbar(label='bikes moved (+: loc1→loc2)')
    plt.xlabel('bikes at location 2')
    plt.ylabel('bikes at location 1')
    plt.title(title)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    V_basic, policy_basic = policy_iteration(expected_return_basic, "PART 1: BASIC")
    print("Basic policy samples:")
    print("policy(10,10) =", policy_basic[10,10])
    print("policy(0,20)  =", policy_basic[0,20])
    print("policy(20,0)  =", policy_basic[20,0])
    plot_policy(policy_basic, "Basic policy")

    V_mod, policy_mod = policy_iteration(expected_return_modified, "PART 2: MODIFIED")
    print("Modified policy samples:")
    print("policy(10,10) =", policy_mod[10,10])
    print("policy(0,20)  =", policy_mod[0,20])
    print("policy(20,0)  =", policy_mod[20,0])
    plot_policy(policy_mod, "Modified policy (free shuttle + parking cost)")

    diff_count = np.sum(policy_basic != policy_mod)
    print("\nStates with different actions:", diff_count, "/", (MAX_BIKES+1)**2)
    print("Average value difference:", float(np.mean(V_mod - V_basic)))