def get_successors(state):
    successors = []
    empty_idx = state.index('_')
    moves = []
    if empty_idx > 0:
        moves.append(empty_idx - 1)
    if empty_idx > 1:
        moves.append(empty_idx - 2)
    if empty_idx < len(state) - 1:
        moves.append(empty_idx + 1)
    if empty_idx < len(state) - 2:
        moves.append(empty_idx + 2)
    
    for idx in moves:
        if 0 <= idx < len(state):
            new_state = state.copy()
            if (state[idx] == 'E' and idx < empty_idx) or (state[idx] == 'W' and idx > empty_idx):
                new_state[idx], new_state[empty_idx] = new_state[empty_idx], new_state[idx]
                move_desc = f"{state[idx]} at {idx} to {empty_idx}"
                successors.append((new_state, move_desc))
    return successors

def dfs_rabbit_leap(state, goal_state, path, visited, depth_limit=100):
    if state == goal_state:
        return path
    if len(path) >= depth_limit:
        return None
    state_tuple = tuple(state)
    if state_tuple in visited:
        return None
    visited.add(state_tuple)
    
    for next_state, move in get_successors(state):
        result = dfs_rabbit_leap(next_state, goal_state, path + [move], visited)
        if result:
            return result
    return None

def print_solution():
    initial_state = ['E', 'E', 'E', '_', 'W', 'W', 'W']
    goal_state = ['W', 'W', 'W', '_', 'E', 'E', 'E']
    solution = dfs_rabbit_leap(initial_state, goal_state, [], set())
    if solution:
        print("DFS Solution found in", len(solution), "steps:")
        state = initial_state.copy()
        print("Initial state:", ''.join(state))
        for i, move in enumerate(solution, 1):
            print(f"Step {i}: {move} -> State: ", end='')
            idx_from = int(move.split(' at ')[1].split(' to ')[0])
            idx_to = int(move.split(' to ')[1])
            state[idx_from], state[idx_to] = state[idx_to], state[idx_from]
            print(''.join(state))
    else:
        print("No solution found with DFS.")

print_solution()