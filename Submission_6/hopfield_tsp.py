import numpy as np
from math import inf
import warnings

try:
    from scipy.special import expit as scipy_expit
    HAS_EXPIT = True
except Exception:
    scipy_expit = None
    HAS_EXPIT = False

try:
    from scipy.optimize import linear_sum_assignment
    HAS_HUNGARIAN = True
except Exception:
    linear_sum_assignment = None
    HAS_HUNGARIAN = False

def stable_sigmoid(x, gain=1.0):
    z = gain * x
    if HAS_EXPIT:
        return scipy_expit(z)
    z_clipped = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z_clipped))


def pairwise_dist(coords):
    N = coords.shape[0]
    d = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            d[i,j] = np.linalg.norm(coords[i]-coords[j])
    return d

def energy_tsp(v, d, A, B, C):
    N = v.shape[0]
    E = 0.5 * A * np.sum((v.sum(axis=0)-1.0)**2) + 0.5 * B * np.sum((v.sum(axis=1)-1.0)**2)
    L = 0.0
    for t in range(N):
        tnext = (t+1) % N
        L += 0.5 * C * np.sum(d * np.outer(v[:,t], v[:,tnext]))
    return E + L

def run_hopfield_tsp(coords,
                     A=400.0, B=400.0, C=1.0,
                     dt=0.02, tau=1.0, gain=8.0,
                     steps=4000, normalize_distances=True,
                     verbose=False):
    N = coords.shape[0]
    d = pairwise_dist(coords)
    if normalize_distances:
        maxd = d.max() if d.max() > 0 else 1.0
        d = d / maxd
    u = 0.05 * np.random.randn(N,N)

    for step in range(steps):
        v = stable_sigmoid(u, gain=gain)

        row_sum = v.sum(axis=1)  
        col_sum = v.sum(axis=0)  

        grad = np.zeros_like(v)
        grad += (B * (row_sum - 1.0))[:, None]
        grad += (A * (col_sum - 1.0))[None, :]

        for t in range(N):
            tnext = (t + 1) % N
            tprev = (t - 1) % N
            grad[:, t] += C * (d.dot(v[:, tnext]) + d.T.dot(v[:, tprev]))

        du = (-u - grad) * (dt / tau)
        u += du

        if step % 200 == 0:
            u = np.clip(u, -50.0/gain, 50.0/gain)

        if verbose and (step % (steps // 4 if steps>=4 else 1) == 0):
            Ev = stable_sigmoid(u, gain=gain)
            E = energy_tsp(Ev, d, A, B, C)
            print(f"[step {step}] energy (approx) = {E:.4f}")

    v = stable_sigmoid(u, gain=gain)
    return v, d

def extract_tour_from_v(v):
    N = v.shape[0]
    tour = np.argmax(v, axis=0).tolist()
    assigned = set()
    final_tour = [-1]*N
    for t in range(N):
        city = tour[t]
        if city not in assigned:
            final_tour[t] = city
            assigned.add(city)
    unassigned = [i for i in range(N) if i not in assigned]
    for t in range(N):
        if final_tour[t] == -1:
            choices = sorted(unassigned, key=lambda c: -v[c,t])
            chosen = choices[0]
            final_tour[t] = chosen
            unassigned.remove(chosen)
    return final_tour

def tour_length(tour, d):
    N = len(tour)
    L = 0.0
    for t in range(N):
        L += d[tour[t], tour[(t+1)%N]]
    return L

def postprocess_with_hungarian(v, d, use_hungarian=HAS_HUNGARIAN):
    if not use_hungarian:
        return None
    cost = -v.T  
    row_ind, col_ind = linear_sum_assignment(cost)
    tour = [None]*v.shape[0]
    for time_idx, city_idx in zip(row_ind, col_ind):
        tour[time_idx] = int(city_idx)
    return tour

if __name__ == "__main__":
    np.random.seed(2)
    N = 10
    coords = np.random.rand(N, 2) * 100.0

    best_tour = None
    best_len = inf
    best_v = None

    restarts = 8
    for r in range(restarts):
        v, d = run_hopfield_tsp(coords,
                                A=400.0, B=400.0, C=0.6,
                                dt=0.02, gain=7.0, steps=5000,
                                normalize_distances=True,
                                verbose=False)
        tour_hung = postprocess_with_hungarian(v, d, use_hungarian=True)
        if tour_hung is not None:
            L = tour_length(tour_hung, d)
            candidate = tour_hung
        else:
            candidate = extract_tour_from_v(v)
            L = tour_length(candidate, d)

        print(f"Restart {r}: length={L:.3f}")
        if L < best_len:
            best_len = L
            best_tour = candidate
            best_v = v

    print("Best length:", best_len)
    print("Best tour:", best_tour)
    M = N*N
    print("Unique pairwise weights (off-diag):", (M*(M-1))//2)
