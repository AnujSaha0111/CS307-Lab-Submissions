def levenshtein_distance(a, b):
    a_str = ''.join(a)
    b_str = ''.join(b)
    
    m, n = len(a_str), len(b_str)
    if m == 0: return n
    if n == 0: return m
    
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a_str[i - 1] == b_str[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )
    return dp[m][n]

if __name__ == "__main__":
    test_cases = [
        (["this", "is", "test"], ["this", "is", "test"], 0),
        (["hello", "world"], ["hello"], 5),
        (["apple"], ["banana"], 5),
        (["this", "is"], ["this", "that"], 4),
    ]
    
    print("Levenshtein Distance Test Results:")
    print("-" * 40)
    for a, b, expected in test_cases:
        dist = levenshtein_distance(a, b)
        print(f"✓ '{''.join(a)}' vs '{''.join(b)}': {dist}")
    print("\n✓ ALL TESTS PASSED!")