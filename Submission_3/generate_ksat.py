import random
import sys

def generate_k_sat(k, m, n):
    formula = []
    for _ in range(m):
        vars_ = random.sample(range(1, n+1), k) 
        clause = [random.choice([-1, 1]) * v for v in vars_]
        formula.append(clause)
    return formula

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python generate_ksat.py <k> <m> <n>")
        sys.exit(1)
    k = int(sys.argv[1])
    m = int(sys.argv[2])
    n = int(sys.argv[3])
    formula = generate_k_sat(k, m, n)
    print(formula)