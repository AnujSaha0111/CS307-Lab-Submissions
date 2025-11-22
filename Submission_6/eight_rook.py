import numpy as np

def energy(v, A=8.0, B=8.0):
    row_sums = v.sum(axis=1)
    col_sums = v.sum(axis=0)
    E = 0.5 * A * np.sum((row_sums - 1.0)**2) + 0.5 * B * np.sum((col_sums - 1.0)**2)
    return E

def try_flip_delta(v, i, j, A=8.0, B=8.0):
    old = v[i, j]
    new = 1 - old
    rsum = v[i, :].sum()
    csum = v[:, j].sum()
    E_before = 0.5 * A * (rsum - 1)**2 + 0.5 * B * (csum - 1)**2
    rsum_new = rsum - old + new
    csum_new = csum - old + new
    E_after = 0.5 * A * (rsum_new - 1)**2 + 0.5 * B * (csum_new - 1)**2
    return E_after - E_before

def solve_eight_rook(max_iters=20000, A=8.0, B=8.0, verbose=False):
    v = np.zeros((8,8), dtype=int)
    for i in range(8):
        j = np.random.randint(8)
        v[i, j] = 1
    E = energy(v, A, B)
    if verbose:
        print("Init energy:", E)
    for it in range(max_iters):
        i = np.random.randint(8)
        j = np.random.randint(8)
        delta = try_flip_delta(v, i, j, A, B)
        if delta < -1e-9:
            v[i,j] = 1 - v[i,j]
            E += delta
        if np.all(v.sum(axis=1) == 1) and np.all(v.sum(axis=0) == 1):
            if verbose:
                print("Found solution iter", it, "E:", energy(v,A,B))
            return v
    if verbose:
        print("No perfect solution after max iterations. E:", energy(v,A,B))
    return v

if __name__ == "__main__":
    np.random.seed(1)
    sol = solve_eight_rook(verbose=True)
    print(sol)
    print("Row sums:", sol.sum(axis=1))
    print("Col sums:", sol.sum(axis=0))
