from preprocessing import preprocess_documents
from astar_search import a_star_alignment
from plagiarism_detector import detect_plagiarism

def plagiarism_detection_system(doc1, doc2, similarity_threshold=0.8): 
    sents1, sents2 = preprocess_documents(doc1, doc2)
    
    path, total_cost = a_star_alignment(sents1, sents2)
    
    if path is None:
        return "No alignment found.", []
    
    plag_pairs = detect_plagiarism(sents1, sents2, path, similarity_threshold)
    
    return total_cost, plag_pairs

def print_results(total_cost, plag_pairs):
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Total Alignment Cost: {total_cost}")
    print(f"Potential Plagiarism Cases: {len(plag_pairs)}")
    print()
    
    if plag_pairs:
        print("DETECTED PLAGIARISM:")
        print("-" * 40)
        for i, pair in enumerate(plag_pairs, 1):
            print(f"{i}. Sentence {pair['sent1_index']} ↔ {pair['sent2_index']}")
            print(f"   Similarity: {pair['similarity']:.3f}")
            print(f"   Doc1: {pair['sent1']}")
            print(f"   Doc2: {pair['sent2']}")
            print()
    else:
        print("✓ No plagiarism detected!")

if __name__ == "__main__":
    print("Plagiarism Detection System")
    print("=" * 40)
    
    print("Enter Document 1:")
    doc1 = input("> ")
    
    print("\nEnter Document 2:")
    doc2 = input("> ")
    
    total_cost, plag_pairs = plagiarism_detection_system(doc1, doc2)
    print_results(total_cost, plag_pairs)