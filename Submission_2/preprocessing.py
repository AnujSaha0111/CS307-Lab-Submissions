import re

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s.!?]', '', text)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    sents = []
    for sent in sentences:
        clean_sent = re.sub(r'[.!?]+$', '', sent).strip()
        words = clean_sent.split()
        if words:
            sents.append(words)
    return sents

def preprocess_documents(doc1, doc2):
    return preprocess_text(doc1), preprocess_text(doc2)

if __name__ == "__main__":
    doc1 = "This is the first sentence. This is the second. Third sentence here!"
    doc2 = "This is first sentence. Second is different. Another one."
    
    sents1, sents2 = preprocess_documents(doc1, doc2)
    
    print("Document 1 sentences:")
    for i, sent in enumerate(sents1):
        print(f"  {i}: {' '.join(sent)}")
    
    print("\nDocument 2 sentences:")
    for i, sent in enumerate(sents2):
        print(f"  {i}: {' '.join(sent)}")