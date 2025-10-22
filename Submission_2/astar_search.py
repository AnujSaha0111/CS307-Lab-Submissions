import heapq

def get_successors(i, j, sents1, sents2):
    """Get possible next states and costs."""
    succ = []
    m, n = len(sents1), len(sents2)
    
    if i < m and j < n:
        from edit_distance import levenshtein_distance
        cost = levenshtein_distance(sents1[i], sents2[j])
        succ.append(((i + 1, j + 1), 'align', cost))
    
    if i < m:
        cost = len(sents1[i])
        succ.append(((i + 1, j), 'delete', cost))
    
    if j < n:
        cost = len(sents2[j])
        succ.append(((i, j + 1), 'insert', cost))
    
    return succ

def precompute_remaining_words(sents):
    """Precompute sum of word lengths from each position to the end."""
    m = len(sents)
    sum_words = [0] * (m + 1)
    for k in range(m - 1, -1, -1):
        sum_words[k] = sum_words[k + 1] + len(sents[k])
    return sum_words

def a_star_alignment(sents1, sents2):
    """Perform A* search for optimal alignment."""
    m, n = len(sents1), len(sents2)
    sum1 = precompute_remaining_words(sents1)
    sum2 = precompute_remaining_words(sents2)
    
    def heuristic(i, j):
        return max(sum1[i], sum2[j])
    
    frontier = []
    heapq.heappush(frontier, (0, 0, 0, 0))  # f, g, i, j
    came_from = {}
    cost_so_far = {}
    came_from[(0, 0)] = None
    cost_so_far[(0, 0)] = 0
    
    while frontier:
        _, g, i, j = heapq.heappop(frontier)
        if i == m and j == n:
            break
        
        for (ni, nj), action, add_cost in get_successors(i, j, sents1, sents2):
            new_g = g + add_cost
            state = (ni, nj)
            if state not in cost_so_far or new_g < cost_so_far[state]:
                cost_so_far[state] = new_g
                f = new_g + heuristic(ni, nj)
                heapq.heappush(frontier, (f, new_g, ni, nj))
                came_from[state] = (i, j, action)
    
    if (m, n) not in came_from:
        return None, None
    
    path = []
    current = (m, n)
    total_cost = cost_so_far[(m, n)]
    
    while current != (0, 0):
        prev_i, prev_j, action = came_from[current]
        
        if action == 'align':
            from edit_distance import levenshtein_distance
            dist = levenshtein_distance(sents1[prev_i], sents2[prev_j])
            path.append((prev_i, prev_j, action, dist))
        elif action == 'delete':
            dist = len(sents1[prev_i])
            path.append((prev_i, None, action, dist))
        elif action == 'insert':
            dist = len(sents2[prev_j])
            path.append((None, prev_j, action, dist))
        
        current = (prev_i, prev_j)
    
    path.reverse()
    return path, total_cost

if __name__ == "__main__":
    from preprocessing import preprocess_text
    
    doc1 = "This is first. This is second."
    doc2 = "This is first. Second is different."
    
    sents1 = preprocess_text(doc1)
    sents2 = preprocess_text(doc2)
    
    path, total_cost = a_star_alignment(sents1, sents2)
    
    print("A* Search Results:")
    print(f"Total Cost: {total_cost}")
    print("Alignment Path:")
    for step in path:
        print(f"  {step}")