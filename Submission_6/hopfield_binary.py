import numpy as np

class HopfieldBinary:
    def __init__(self, N):
        self.N = N
        self.W = np.zeros((N, N), dtype=float)

    def store_patterns(self, patterns):
        N = self.N
        W = np.zeros((N, N), dtype=float)
        for p in patterns:
            W += np.outer(p, p)
        np.fill_diagonal(W, 0.0)
        self.W = W / N

    def recall(self, s_init, max_iters=1000, asynchronous=True):
        s = s_init.copy()
        N = self.N
        if asynchronous:
            for it in range(max_iters):
                changed = False
                for i in np.random.permutation(N):
                    h = np.dot(self.W[i], s)
                    s_new = 1 if h >= 0 else -1
                    if s_new != s[i]:
                        s[i] = s_new
                        changed = True
                if not changed:
                    break
        else:
            for _ in range(max_iters):
                h = self.W.dot(s)
                s_new = np.where(h >= 0, 1, -1)
                if np.array_equal(s_new, s):
                    break
                s = s_new
        return s

def random_patterns(N, P):
    return [np.where(np.random.rand(N) > 0.5, 1, -1) for _ in range(P)]

def flip_bits(vec, nflip):
    v = vec.copy()
    idx = np.random.choice(len(v), size=nflip, replace=False)
    v[idx] *= -1
    return v

def capacity_experiment(N=100, trials=20, store_list=None, noise_frac=0.05):
    if store_list is None:
        store_list = [2,5,8,10,12,14,16,18]
    results = {}
    for P in store_list:
        succ = 0
        for t in range(trials):
            patterns = random_patterns(N, P)
            hop = HopfieldBinary(N)
            hop.store_patterns(patterns)
            pat = patterns[np.random.randint(P)]
            noisy = flip_bits(pat, int(noise_frac * N))
            rec = hop.recall(noisy)
            if np.array_equal(rec, pat):
                succ += 1
        results[P] = succ / trials
    return results

if __name__ == "__main__":
    np.random.seed(0)
    N = 100
    print("Theoretical capacity ≈", 0.138 * N)
    res = capacity_experiment(N=N, trials=40, store_list=[2,5,8,10,12,13,14,15], noise_frac=0.08)
    for P, acc in res.items():
        print(f"P={P}, recall success (8% flips): {acc:.2f}")
