import random
from itertools import combinations
from heuristics import h1
def vnd(formula, n, h_func, max_tries=10):
    k_max = 3
    steps = 0
    for t in range(max_tries):
        assignment = [None] + [random.choice([False, True]) for _ in range(n)]
        k = 1
        while k <= k_max:
            steps += 1
            if h1(formula, assignment) == 0:
                return assignment, steps, True
            current_h = h_func(formula, assignment)
            best_h = current_h
            best_flips = None
            for flips in combinations(range(1, n+1), k):
                for v in flips:
                    assignment[v] = not assignment[v]
                new_h = h_func(formula, assignment)
                if new_h < best_h:
                    best_h = new_h
                    best_flips = flips
                for v in flips:
                    assignment[v] = not assignment[v]
            if best_h < current_h:
                for v in best_flips:
                    assignment[v] = not assignment[v]
                k = 1
            else:
                k = k + 1
    return None, steps, False