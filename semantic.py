from sentence_transformers import SentenceTransformer, util

# Load the pre-trained Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')


def calculate_semantic_similarity(text1, text2):
    # Encode the sentences
    embeddings = model.encode([text1, text2])

    # Compute cosine similarity
    cosine_similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])

    return cosine_similarity.item()


# Example usage
text1 = "The quick brown fox jumps over the lazy dog."
text2 = "A fast brown fox leaps over a lazy dog."

similarity_score = calculate_semantic_similarity(text1, text2)
print(f"Semantic Similarity: {similarity_score}")
