import numpy as np
import random
import math
import matplotlib.pyplot as plt
import os

INPUT_PATH = "scrambled_lena.mat"
NUM_TILES = 4
TRANSFORMS_PER_TILE = 8
RANDOM_SEED = 42
INITIAL_T = 5000.0
MIN_T = 1e-6
ALPHA = 0.9995
MAX_ITER = 1200000
EDGE_COMPARE_WIDTH = 2

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def load_image_flexible(path):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, path)

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"File not found at {full_path}")

    try:
        data = np.loadtxt(full_path, dtype=np.uint8, skiprows=5)
        arr = np.asarray(data)
        n = int(np.sqrt(arr.size))
        if n * n != arr.size:
            raise ValueError("Loaded text does not form a square image.")
        img = arr.reshape(n, n)
        return img
    except Exception as e:
        print(f"Failed to load as text: {e}")
        raise

scrambled = load_image_flexible(INPUT_PATH)
if scrambled.ndim != 2:
    raise ValueError("Loaded image must be grayscale (2D).")
img_size = scrambled.shape[0]
if scrambled.shape[0] != scrambled.shape[1]:
    raise ValueError("Image must be square.")
if img_size % NUM_TILES != 0:
    raise ValueError(f"Image size {img_size} not divisible by NUM_TILES={NUM_TILES}")

tile_size = img_size // NUM_TILES
print(f"Loaded image {img_size}x{img_size}, using {NUM_TILES}x{NUM_TILES} grid, tile size {tile_size}x{tile_size}")

def transform_identity(x): return x
def transform_rot90(x): return np.rot90(x, 1)
def transform_rot180(x): return np.rot90(x, 2)
def transform_rot270(x): return np.rot90(x, 3)
def transform_flip_ud(x): return np.flipud(x)
def transform_flip_lr(x): return np.fliplr(x)
def transform_rot90_flip_ud(x): return np.flipud(np.rot90(x, 1))
def transform_rot90_flip_lr(x): return np.fliplr(np.rot90(x, 1))

transformations = [
    transform_identity, transform_rot90, transform_rot180, transform_rot270,
    transform_flip_ud, transform_flip_lr, transform_rot90_flip_ud, transform_rot90_flip_lr
]

tiles = []
original_tile_count = NUM_TILES * NUM_TILES
for i in range(NUM_TILES):
    for j in range(NUM_TILES):
        base = scrambled[i * tile_size:(i + 1) * tile_size, j * tile_size:(j + 1) * tile_size]
        for f in transformations:
            tiles.append(f(base).copy())
assert len(tiles) == original_tile_count * TRANSFORMS_PER_TILE

grid = (np.arange(original_tile_count) * TRANSFORMS_PER_TILE).reshape(NUM_TILES, NUM_TILES)
print("Initial grid shape:", grid.shape)

def tile_gradient(t):
    gx = np.abs(np.gradient(t.astype(np.float32), axis=1))
    gy = np.abs(np.gradient(t.astype(np.float32), axis=0))
    return gx + gy

tile_gradients = [tile_gradient(t) for t in tiles]

def compute_energy(grid, tiles, grads, tile_size, num_tiles, transforms_per_tile, width=EDGE_COMPARE_WIDTH):
    energy = 0.0
    for i in range(num_tiles):
        for j in range(num_tiles - 1):
            left_val = grid[i, j]
            right_val = grid[i, j + 1]
            left_tile = tiles[left_val]
            right_tile = tiles[right_val]
            left_grad = grads[left_val]
            right_grad = grads[right_val]
            left_cols = left_tile[:, -width:].astype(np.int32)
            right_cols = right_tile[:, :width].astype(np.int32)
            energy += np.sum(np.abs(left_cols - right_cols))
            left_g = left_grad[:, -width:]
            right_g = right_grad[:, :width]
            energy += 0.5 * np.sum(np.abs(left_g - right_g))
    for i in range(num_tiles - 1):
        for j in range(num_tiles):
            top_val = grid[i, j]
            bot_val = grid[i + 1, j]
            top_tile = tiles[top_val]
            bot_tile = tiles[bot_val]
            top_grad = grads[top_val]
            bot_grad = grads[bot_val]
            top_rows = top_tile[-width:, :].astype(np.int32)
            bot_rows = bot_tile[:width, :].astype(np.int32)
            energy += np.sum(np.abs(top_rows - bot_rows))
            top_g = top_grad[-width:, :]
            bot_g = bot_grad[:width, :]
            energy += 0.5 * np.sum(np.abs(top_g - bot_g))
    return energy

def simulated_annealing(grid, tiles, grads, tile_size, num_tiles, transforms_per_tile,
                        initial_T=INITIAL_T, min_T=MIN_T, alpha=ALPHA, max_iter=MAX_ITER):
    current_grid = grid.copy()
    current_energy = compute_energy(current_grid, tiles, grads, tile_size, num_tiles, transforms_per_tile)
    T = initial_T
    best_grid = current_grid.copy()
    best_energy = current_energy

    for it in range(1, max_iter + 1):
        move_type = random.choices(
            ["swap_positions", "swap_transforms", "change_transform", "rotate_transform"],
            weights=[0.5, 0.15, 0.25, 0.10], k=1
        )[0]
        undo_info = None

        if move_type == "swap_positions":
            i1, j1 = random.randrange(num_tiles), random.randrange(num_tiles)
            i2, j2 = random.randrange(num_tiles), random.randrange(num_tiles)
            if i1 == i2 and j1 == j2: continue
            a = current_grid[i1, j1]; b = current_grid[i2, j2]
            current_grid[i1, j1], current_grid[i2, j2] = b, a
            undo_info = ("swap_positions", (i1, j1, i2, j2, a, b))
        elif move_type == "swap_transforms":
            i1, j1 = random.randrange(num_tiles), random.randrange(num_tiles)
            i2, j2 = random.randrange(num_tiles), random.randrange(num_tiles)
            v1 = current_grid[i1, j1]; v2 = current_grid[i2, j2]
            b1 = v1 // transforms_per_tile; t1 = v1 % transforms_per_tile
            b2 = v2 // transforms_per_tile; t2 = v2 % transforms_per_tile
            current_grid[i1, j1] = b1 * transforms_per_tile + t2
            current_grid[i2, j2] = b2 * transforms_per_tile + t1
            undo_info = ("swap_transforms", (i1, j1, i2, j2, v1, v2))
        elif move_type == "change_transform":
            i, j = random.randrange(num_tiles), random.randrange(num_tiles)
            old = int(current_grid[i, j])
            base_idx = old // transforms_per_tile
            new_t = random.randrange(transforms_per_tile)
            new_val = base_idx * transforms_per_tile + new_t
            current_grid[i, j] = new_val
            undo_info = ("change_transform", (i, j, old))
        else:  
            i, j = random.randrange(num_tiles), random.randrange(num_tiles)
            old = int(current_grid[i, j])
            base_idx = old // transforms_per_tile
            cur_t = old % transforms_per_tile
            new_t = (cur_t + random.choice([1, -1])) % transforms_per_tile
            new_val = base_idx * transforms_per_tile + new_t
            current_grid[i, j] = new_val
            undo_info = ("change_transform", (i, j, old))

        new_energy = compute_energy(current_grid, tiles, grads, tile_size, num_tiles, transforms_per_tile)
        delta = new_energy - current_energy
        accept = False
        if delta <= 0:
            accept = True
        else:
            if T > 0 and random.random() < math.exp(-delta / T):
                accept = True

        if accept:
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_grid = current_grid.copy()
        else:
            typ, info = undo_info
            if typ == "swap_positions":
                i1, j1, i2, j2, a, b = info
                current_grid[i1, j1], current_grid[i2, j2] = a, b
            elif typ == "swap_transforms":
                i1, j1, i2, j2, v1, v2 = info
                current_grid[i1, j1], current_grid[i2, j2] = v1, v2
            elif typ == "change_transform":
                i, j, old = info
                current_grid[i, j] = old

        T *= alpha
        if T < min_T:
            break

    return best_grid, best_energy

best_grid, best_energy = simulated_annealing(grid, tiles, tile_gradients, tile_size, NUM_TILES, TRANSFORMS_PER_TILE)
print(f"Solved with energy: {best_energy} after {MAX_ITER} iterations")

unscrambled = np.zeros_like(scrambled, dtype=np.uint8)
for i in range(NUM_TILES):
    for j in range(NUM_TILES):
        val = int(best_grid[i, j])
        tile = tiles[val]
        unscrambled[i * tile_size:(i + 1) * tile_size, j * tile_size:(j + 1) * tile_size] = tile

plt.imshow(unscrambled, cmap='gray')
plt.axis('off')
plt.show()