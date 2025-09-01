from collections import deque

def get_successors(state):
    successors = []
    empty_idx = state.index('_')
    # Possible moves: 1 step left/right or 2 steps left/right (jumping over one rabbit)
    moves = []
    if empty_idx > 0:  # Can move a rabbit from left to empty spot
        moves.append(empty_idx - 1)  # One step left
    if empty_idx > 1:  # Can jump a rabbit from two steps left
        moves.append(empty_idx - 2)
    if empty_idx < len(state) - 1:  # Can move a rabbit from right
        moves.append(empty_idx + 1)
    if empty_idx < len(state) - 2:  # Can jump a rabbit from two steps right
        moves.append(empty_idx + 2)
    
    for idx in moves:
        if 0 <= idx < len(state):
            new_state = state.copy()
            # Check if move is valid (E moves right, W moves left)
            if (state[idx] == 'E' and idx < empty_idx) or (state[idx] == 'W' and idx > empty_idx):
                new_state[idx], new_state[empty_idx] = new_state[empty_idx], new_state[idx]
                move_desc = f"{state[idx]} at {idx} to {empty_idx}"
                successors.append((new_state, move_desc))
    return successors

def bfs_rabbit_leap():
    initial_state = ['E', 'E', 'E', '_', 'W', 'W', 'W']
    goal_state = ['W', 'W', 'W', '_', 'E', 'E', 'E']
    queue = deque([(initial_state, [])])
    visited = set()
    visited.add(tuple(initial_state))
    
    while queue:
        state, path = queue.popleft()
        if state == goal_state:
            return path
        for next_state, move in get_successors(state):
            state_tuple = tuple(next_state)
            if state_tuple not in visited:
                visited.add(state_tuple)
                queue.append((next_state, path + [move]))
    return None

def print_solution():
    solution = bfs_rabbit_leap()
    if solution:
        print("BFS Solution found in", len(solution), "steps:")
        state = ['E', 'E', 'E', '_', 'W', 'W', 'W']
        print("Initial state:", ''.join(state))
        for i, move in enumerate(solution, 1):
            print(f"Step {i}: {move} -> State: ", end='')
            # Reconstruct state for display
            idx_from = int(move.split(' at ')[1].split(' to ')[0])
            idx_to = int(move.split(' to ')[1])
            state[idx_from], state[idx_to] = state[idx_to], state[idx_from]
            print(''.join(state))
    else:
        print("No solution found with BFS.")

print_solution()