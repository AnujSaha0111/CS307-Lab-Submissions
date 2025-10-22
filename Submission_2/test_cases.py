def run_test_case(name, doc1, doc2, expected_plagiarism_count):
    from plagiarism_detection import plagiarism_detection_system
    
    total_cost, plag_pairs = plagiarism_detection_system(doc1, doc2)
    
    print(f"\n{'='*60}")
    print(f"TEST CASE: {name}")
    print(f"{'='*60}")
    print(f"Total Alignment Cost: {total_cost}")
    print(f"Plagiarism Pairs Found: {len(plag_pairs)}")
    print(f"Expected: {expected_plagiarism_count}")
    
    status = "✓ PASSED" if len(plag_pairs) == expected_plagiarism_count else "✗ FAILED"
    print(f"Status: {status}")
    
    for pair in plag_pairs:
        print(f"  - Sentence {pair['sent1_index']}: {pair['similarity']:.3f}")

def run_all_tests():
    """Run all test cases."""
    
    # Test Case 1: Identical Documents
    doc1_1 = "This is sentence one. This is sentence two. This is three."
    doc2_1 = "This is sentence one. This is sentence two. This is three."
    run_test_case("1. Identical Documents", doc1_1, doc2_1, 3)
    
    # Test Case 2: Slightly Modified
    doc1_2 = "This is first sentence. Second sentence here. Third one."
    doc2_2 = "This is first sentence. Second sentence changed. Third one."
    run_test_case("2. Slightly Modified", doc1_2, doc2_2, 2)
    
    # Test Case 3: Completely Different
    doc1_3 = "Cat dog bird. House car tree. Apple banana."
    doc2_3 = "Red blue green. Mountain river sea. Book pen paper."
    run_test_case("3. Completely Different", doc1_3, doc2_3, 0)
    
    # Test Case 4: Partial Overlap
    doc1_4 = "First sentence. Second here. Third different. Fourth unique."
    doc2_4 = "First sentence. Second here. Totally new. Another one."
    run_test_case("4. Partial Overlap", doc1_4, doc2_4, 2)

if __name__ == "__main__":
    print("Running Plagiarism Detection Test Cases")
    print("=" * 60)
    run_all_tests()
    print("\n" + "="*60)