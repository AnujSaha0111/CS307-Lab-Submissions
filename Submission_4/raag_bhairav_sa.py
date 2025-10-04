import random
import math

notes = ["Sa", "Re(b)", "Ga", "Ma", "Pa", "Dha(b)", "Ni", "Sa'"]
note_to_index = {note: i for i, note in enumerate(notes)}
pakad = ["Re(b)", "Sa", "Pa", "Ma", "Re(b)"]

def has_pakad(melody):
    for i in range(len(melody) - len(pakad) + 1):
        if melody[i:i + len(pakad)] == pakad:
            return True
    return False

def energy(melody):
    score = 0.0
    for i in range(len(melody) - 1):
        d = abs(note_to_index[melody[i]] - note_to_index[melody[i + 1]])
        score += d ** 2
    for i in range(len(melody) - 2):
        if melody[i] == melody[i + 1] == melody[i + 2]:
            score += 20 
    if not has_pakad(melody):
        score += 10000
    return score

def generate_bhairav_melody_sa(length=16):
    melody = pakad[:]
    while len(melody) < length:
        melody.append(random.choice(notes))
    
    T = 1000.0
    min_T = 0.1
    cool_rate = 0.99
    current_energy = energy(melody)
    best_melody = list(melody)
    best_energy = current_energy
    
    iteration = 0
    max_iterations = 10000
    
    while T > min_T and iteration < max_iterations:
        new_melody = list(melody)
        pos = random.randint(0, length - 1)
        old_note = new_melody[pos]
        new_note = random.choice(notes)
        while new_note == old_note:
            new_note = random.choice(notes)
        new_melody[pos] = new_note
        
        new_e = energy(new_melody)
        delta = new_e - current_energy
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            melody = new_melody
            current_energy = new_e
            if current_energy < best_energy:
                best_energy = current_energy
                best_melody = list(melody)
        
        T *= cool_rate
        iteration += 1
    
    print(f"Generated with energy: {best_energy} after {iteration} iterations")
    return best_melody

melody = generate_bhairav_melody_sa()
print("RAAG BHAIRAV MELODY:")
print(" - ".join(melody))