import random

class MENACE:
    def __init__(self):
        self.boxes = {}
        self.ordered_boxes = [[], [], [], [], []] 
        self.start = [8, 4, 2, 1]
        self.incentives = [1, 3, -1] 
        self.moves = []

        self.rotations = [
            [0,1,2,3,4,5,6,7,8],
            [0,3,6,1,4,7,2,5,8],
            [6,3,0,7,4,1,8,5,2],
            [6,7,8,3,4,5,0,1,2],
            [8,7,6,5,4,3,2,1,0],
            [8,5,2,7,4,1,6,3,0],
            [2,5,8,1,4,7,0,3,6],
            [2,1,0,5,4,3,8,7,6]
        ]

    def array_fill(self, length, value):
        return [value] * length

    def count(self, arr, value):
        return sum(1 for x in arr if x == value)

    def apply_rotation(self, pos, rot):
        return "".join(pos[rot[i]] for i in range(9))

    def find_all_rotations(self, pos):
        max_val = ""
        max_rots = []
        for i, rot in enumerate(self.rotations):
            candidate = self.apply_rotation(pos, rot)
            if candidate > max_val:
                max_val = candidate
                max_rots = []
            if candidate == max_val:
                max_rots.append(i)
        return max_rots

    def rotation_is_max(self, pos):
        return self.find_all_rotations(pos)[0] == 0

    def new_box(self, pos, start):
        bead_index = min(len(self.start) - 1, int((9 - self.count(pos, '0')) / 2))
        start_beads = self.start[bead_index]
        
        box = self.array_fill(9, start_beads)
        for i, c in enumerate(pos):
            if c != '0':
                box[i] = 0
        return box

    def add_box(self, pos, ply):
        self.ordered_boxes[ply].append(pos)
        self.boxes[pos] = self.new_box(pos, self.start) 

    def generate_all_menace_states(self):
        self.boxes = {}
        self.ordered_boxes = [[], [], [], [], []]
        self._search_recursive("000000000")

    def _search_recursive(self, board):
        played = 9 - self.count(board, '0')
        current_player = 1 if played % 2 == 0 else 2
        ply = played // 2 

        if current_player == 1 and self.winner(board) is False and self.rotation_is_max(board):
            self.add_box(board, ply)

        if self.winner(board) is not False or played == 9:
            return

        for i in range(9):
            if board[i] == '0':
                new_board_list = list(board)
                new_board_list[i] = str(current_player)
                new_board_str = "".join(new_board_list)
                self._search_recursive(new_board_str)

    def winner(self, pos):
        pwns = [
            (0,1,2), (3,4,5), (6,7,8),
            (0,3,6), (1,4,7), (2,5,8),
            (0,4,8), (6,4,2)
        ]
        for (x, y, z) in pwns:
            if pos[x] != '0' and pos[x] == pos[y] == pos[z]:
                return int(pos[x])
        if self.count(pos, '0') == 0:
            return 0  # Draw
        return False

    def make_move(self, beads):
        total = sum(beads)
        if total == 0:
            return "resign"
        r = random.randint(0, total-1)
        s = 0
        for i, b in enumerate(beads):
            s += b
            if r < s:
                return i

    def reset(self):
        self.moves = []
        self.generate_all_menace_states() 

    def select_move(self, board):
        pos = "".join(str(x) for x in board)
        max_rots = self.find_all_rotations(pos)
        rot = random.choice(max_rots)
        rotated_pos = self.apply_rotation(pos, self.rotations[rot])
        
        if rotated_pos not in self.boxes:
            return "resign" 
            
        move = self.make_move(self.boxes[rotated_pos])
        if move == "resign":
            return move
            
        inv_move = self.rotations[rot].index(move) 
        self.moves.append((rotated_pos, move))
        return inv_move

    def update_beads(self, result):
        # result: 0=draw, 1=win, 2=loss
        idx = self.incentives[result]
        
        for state, move in self.moves:
            self.boxes[state][move] = max(0, self.boxes[state][move] + idx)
            
        self.moves = []

def print_board(board):
    symbols = [' ', 'X', 'O']
    for i in range(0, 9, 3):
        print("|".join(symbols[board[j]] for j in range(i, i+3)))
    print("-------")

def get_random_move(board):
    free_positions = [i for i, v in enumerate(board) if v == 0]
    if not free_positions:
        return None
    return random.choice(free_positions)

def is_winner(board, player):
    wins = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),
        (0, 3, 6), (1, 4, 7), (2, 5, 8),
        (0, 4, 8), (2, 4, 6)
    ]
    for a, b, c in wins:
        if board[a] == board[b] == board[c] == player:
            return True
    return False

def is_draw(board):
    return all(x != 0 for x in board)

def play_one_game(menace):
    board = [0] * 9
    current_player = 1 # MENACE is always Player 1
    menace.moves = []

    while True:
        if current_player == 1:
            # MENACE's move
            move = menace.select_move(board)
            if move == "resign":
                menace.update_beads(2) # Loss
                return 2
            board[move] = 1
            
            if is_winner(board, 1):
                menace.update_beads(1) # Win
                return 1
            elif is_draw(board):
                menace.update_beads(0) # Draw
                return 0
            current_player = 2
        else:
            # Random Opponent's move
            move = get_random_move(board)
            if move is None:
                menace.update_beads(0) # Draw
                return 0
            board[move] = 2
            
            if is_winner(board, 2):
                menace.update_beads(2) # Loss
                return 2
            elif is_draw(board):
                menace.update_beads(0) # Draw
                return 0
            current_player = 1

def train_menace(games=50000):
    menace = MENACE()
    menace.reset() 
    
    total_states = len(menace.boxes)
    print(f" MENACE successfully initialized with {total_states} unique states.")

    wins = draws = losses = 0
    total_games = games
    block_size = total_games // 10 
    
    print("\n--- Starting Training ---")
    
    for i in range(total_games):
        result = play_one_game(menace)
        if result == 1:
            wins += 1
        elif result == 0:
            draws += 1
        else:
            losses += 1
            
        if (i + 1) % block_size == 0:
            current_games = i + 1
            win_rate = (wins / block_size) * 100
            draw_rate = (draws / block_size) * 100
            loss_rate = (losses / block_size) * 100
            
            print(f"Block {current_games // block_size}: (Games: {current_games}) Wins/Draws/Losses: {wins}/{draws}/{losses}")
            print(f"  Rates in Block: Win: {win_rate:.2f}% | Draw: {draw_rate:.2f}% | Loss: {loss_rate:.2f}%")
            
            wins = draws = losses = 0 
            
    print("\n--- Training Complete ---")

if __name__ == "__main__":
    train_menace(games=50000)