from utils import get_literal_value
def h1(formula, assignment):
    uns = 0
    for clause in formula:
        if not any(get_literal_value(lit, assignment) for lit in clause):
            uns += 1
    return uns
def h2(formula, assignment):
    score = 0.0
    for clause in formula:
        num_true = sum(1 for lit in clause if get_literal_value(lit, assignment))
        if num_true == 0:
            score += 1.0
        else:
            score += (3 - num_true) * 0.1
    return score