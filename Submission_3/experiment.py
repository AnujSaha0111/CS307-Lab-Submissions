from generate_ksat import generate_k_sat
from heuristics import h1, h2
from hill_climbing import hill_climbing
from beam_search import beam_search
from vnd import vnd

def run_experiment(n, m_list, num_instances=5):
    k = 3
    for m in m_list:
        print(f"\nFor n={n}, m={m}")
        for i in range(num_instances):
            formula = generate_k_sat(k, m, n)
            for h, h_name in [(h1, "h1"), (h2, "h2")]:
                _, steps, solved = hill_climbing(formula, n, h)
                N = steps * n
                p = steps / N if N > 0 else 0
                print(f"Instance {i}, {h_name}, Hill-Climbing: solved={solved}, steps={steps}, penetrance={p:.4f}")
                
                _, steps, solved = beam_search(formula, n, 3, h)
                N = steps * 3 * n
                p = steps / N if N > 0 else 0
                print(f"Instance {i}, {h_name}, Beam-3: solved={solved}, steps={steps}, penetrance={p:.4f}")
                
                _, steps, solved = beam_search(formula, n, 4, h)
                N = steps * 4 * n
                p = steps / N if N > 0 else 0
                print(f"Instance {i}, {h_name}, Beam-4: solved={solved}, steps={steps}, penetrance={p:.4f}")
                
                _, steps, solved = vnd(formula, n, h)
                N = steps * (n**3 / 18)
                p = steps / N if N > 0 else 0
                print(f"Instance {i}, {h_name}, VND: solved={solved}, steps={steps}, penetrance={p:.4f}")

if __name__ == "__main__":
    n_list = [10, 20]
    for n in n_list:
        m_list = [int(2 * n), int(4 * n), int(6 * n)]
        run_experiment(n, m_list)