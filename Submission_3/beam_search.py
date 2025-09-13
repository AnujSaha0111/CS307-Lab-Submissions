import random
from heuristics import h1 

def beam_search(formula, n, beam_width, h_func, max_steps=1000):
    beam = [[None] + [random.choice([False, True]) for _ in range(n)] for _ in range(beam_width)]
    for step in range(max_steps):
        for ass in beam:
            if h1(formula, ass) == 0:
                return ass, step + 1, True
        successors = []
        for ass in beam:
            for v in range(1, n+1):
                new_ass = ass.copy()
                new_ass[v] = not new_ass[v]
                successors.append(new_ass)
        successors.sort(key=lambda ass: h_func(formula, ass))
        beam = successors[:beam_width]
    return None, max_steps, False