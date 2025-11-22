import heapq
import re

def preprocess_text(text):
    # Split by sentence delimiters (., !, ?)
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
    # Normalize: lowercase and strip whitespace
    normalized_sentences = [s.lower().strip() for s in sentences if s.strip()]
    return normalized_sentences

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def heuristic(i, j, doc1_sentences, doc2_sentences):
    remaining_doc1 = len(doc1_sentences) - i
    remaining_doc2 = len(doc2_sentences) - j
    return abs(remaining_doc1 - remaining_doc2)

def a_star_alignment(doc1_sentences, doc2_sentences):    
    start_state = (0, 0, 0, 0, []) # f, g, i, j, path
    pq = [start_state]
    visited = set()
    
    rows = len(doc1_sentences)
    cols = len(doc2_sentences)
    
    while pq:
        f, g, i, j, path = heapq.heappop(pq)
        
        if (i, j) in visited:
            continue
        visited.add((i, j))
        
        # Goal state
        if i == rows and j == cols:
            return path, g
            
        # Transitions        
        # 1. Match/Align current sentences
        if i < rows and j < cols:
            cost = levenshtein_distance(doc1_sentences[i], doc2_sentences[j])
            new_g = g + cost
            new_h = heuristic(i + 1, j + 1, doc1_sentences, doc2_sentences)
            new_path = path + [(i, j, cost, 'MATCH')]
            heapq.heappush(pq, (new_g + new_h, new_g, i + 1, j + 1, new_path))            
        # 2. Skip sentence in doc1
        if i < rows:
            cost = len(doc1_sentences[i]) 
            new_g = g + cost
            new_h = heuristic(i + 1, j, doc1_sentences, doc2_sentences)
            new_path = path + [(i, None, cost, 'SKIP1')]
            heapq.heappush(pq, (new_g + new_h, new_g, i + 1, j, new_path))            
        # 3. Skip sentence in doc2
        if j < cols:
            cost = len(doc2_sentences[j])
            new_g = g + cost
            new_h = heuristic(i, j + 1, doc1_sentences, doc2_sentences)
            new_path = path + [(None, j, cost, 'SKIP2')]
            heapq.heappush(pq, (new_g + new_h, new_g, i, j + 1, new_path))
            
    return [], float('inf')

def detect_plagiarism(text1, text2, threshold=10):
    s1 = preprocess_text(text1)
    s2 = preprocess_text(text2)
    
    alignment, total_cost = a_star_alignment(s1, s2)
    
    suspicious_pairs = []
    for item in alignment:
        idx1, idx2, cost, type_ = item
        if type_ == 'MATCH':
            # If cost is low relative to length, it's likely plagiarism
            # Simple threshold check
            if cost <= threshold:
                suspicious_pairs.append({
                    'doc1_idx': idx1,
                    'doc2_idx': idx2,
                    'doc1_sent': s1[idx1],
                    'doc2_sent': s2[idx2],
                    'cost': cost
                })
                
    return {
        'total_cost': total_cost,
        'alignment': alignment,
        'suspicious_pairs': suspicious_pairs
    }

if __name__ == "__main__":
    t1 = "This is a test. It is a good day."
    t2 = "This is a test. It is a bad day."
    result = detect_plagiarism(t1, t2)
    print(f"Total Cost: {result['total_cost']}")
    for pair in result['suspicious_pairs']:
        print(f"Suspicious: '{pair['doc1_sent']}' vs '{pair['doc2_sent']}' (Cost: {pair['cost']})")