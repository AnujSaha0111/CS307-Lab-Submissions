import unittest
from plagiarism_detector import detect_plagiarism

class TestPlagiarismDetector(unittest.TestCase):

    def test_identical_documents(self):
        print("\n Test Case 1: Identical Documents ")
        doc1 = "The quick brown fox jumps over the lazy dog."
        doc2 = "The quick brown fox jumps over the lazy dog."
        result = detect_plagiarism(doc1, doc2)
        print(f"Total Cost: {result['total_cost']}")
        self.assertEqual(result['total_cost'], 0)
        self.assertEqual(len(result['suspicious_pairs']), 1) # 1 sentence

    def test_slightly_modified_document(self):
        print("\n Test Case 2: Slightly Modified Document ")
        doc1 = "The quick brown fox jumps over the lazy dog."
        doc2 = "The fast brown fox leaps over the lazy dog."
        result = detect_plagiarism(doc1, doc2)
        print(f"Total Cost: {result['total_cost']}")
        self.assertTrue(len(result['suspicious_pairs']) > 0)
        self.assertTrue(result['total_cost'] > 0)

    def test_completely_different_documents(self):
        print("\n Test Case 3: Completely Different Documents ")
        doc1 = "The quick brown fox jumps over the lazy dog."
        doc2 = "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
        result = detect_plagiarism(doc1, doc2)
        print(f"Total Cost: {result['total_cost']}")
        # Should be high cost, no suspicious pairs likely if threshold is tight
        # The cost should be high.
        self.assertTrue(result['total_cost'] > 10)

    def test_partial_overlap(self):
        print("\n Test Case 4: Partial Overlap ")
        doc1 = "The quick brown fox jumps over the lazy dog. It is a sunny day."
        doc2 = "It is a sunny day. The quick brown fox jumps over the lazy dog."        
        result = detect_plagiarism(doc1, doc2)
        print(f"Total Cost: {result['total_cost']}")
        self.assertTrue(len(result['suspicious_pairs']) >= 1)

if __name__ == '__main__':
    unittest.main()
