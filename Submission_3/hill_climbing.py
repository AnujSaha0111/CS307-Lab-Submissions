import random
from heuristics import h1
def hill_climbing(formula, n, h_func, max_tries=100, max_flips=1000):
    steps = 0
    for t in range(max_tries):
        assignment = [None] + [random.choice([False, True]) for _ in range(n)]
        for f in range(max_flips):
            steps += 1
            if h1(formula, assignment) == 0:
                return assignment, steps, True
            current_h = h_func(formula, assignment)
            best_h = current_h
            best_vars = []
            for v in range(1, n+1):
                assignment[v] = not assignment[v]
                new_h = h_func(formula, assignment)
                if new_h < best_h:
                    best_h = new_h
                    best_vars = [v]
                elif new_h == best_h:
                    best_vars.append(v)
                assignment[v] = not assignment[v]
            if best_h < current_h:
                v = random.choice(best_vars)
                assignment[v] = not assignment[v]
            else:
                break
    return None, steps, False