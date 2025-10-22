def detect_plagiarism(sents1, sents2, path, threshold=0.8):
    """Detect plagiarism with 80% similarity threshold."""
    plagiarized_pairs = []
    for idx1, idx2, action, dist in path:
        if action == 'align':
            len1 = sum(len(word) for word in sents1[idx1])
            len2 = sum(len(word) for word in sents2[idx2])
            max_len = max(len1, len2, 1)
            similarity = 1 - (dist / max_len)
            if similarity > threshold:
                plagiarized_pairs.append({
                    'sent1_index': idx1,
                    'sent1': ' '.join(sents1[idx1]),
                    'sent2_index': idx2,
                    'sent2': ' '.join(sents2[idx2]),
                    'distance': dist,
                    'similarity': round(similarity, 3)
                })
    return plagiarized_pairs

if __name__ == "__main__":
    from preprocessing import preprocess_text
    from astar_search import a_star_alignment
    
    doc1 = "This is the first sentence. This is the second sentence."
    doc2 = "This is the first sentence. Second sentence is different."
    
    sents1 = preprocess_text(doc1)
    sents2 = preprocess_text(doc2)
    path, _ = a_star_alignment(sents1, sents2)
    
    plag_pairs = detect_plagiarism(sents1, sents2, path)
    
    print("Plagiarism Detection Results:")
    print("-" * 50)
    for pair in plag_pairs:
        print(f"✓ Match {pair['sent1_index']} ↔ {pair['sent2_index']}: {pair['similarity']}")
        print(f"  Doc1: {pair['sent1']}")
        print(f"  Doc2: {pair['sent2']}")
        print()