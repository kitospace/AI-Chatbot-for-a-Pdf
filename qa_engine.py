import numpy as np
from nltk.tokenize import sent_tokenize


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:  # Prevent division by zero
        return 0
    return dot_product / (norm_a * norm_b)


def get_most_relevant_sentence(chunk, query, model):
    sentences = sent_tokenize(chunk)
    print(f"DEBUG: Sentences in chunk: {sentences}")  # Debugging log

    if not sentences:
        return "No sentences found in the chunk."

    sentence_embeddings = model.encode(sentences)
    query_embedding = model.encode([query])[0]
    # print(f"DEBUG: Query embedding: {query_embedding}")  # Debugging log

    # Compute similarities
    similarities = [cosine_similarity(query_embedding, sent_emb) for sent_emb in sentence_embeddings]
    # print(f"DEBUG: Sentence similarities: {similarities}")  # Debugging log

    best_idx = np.argmax(similarities)
    return sentences[best_idx]


def find_best_answer(query, chunks, embeddings, model, threshold=0.15):
    if chunks is None or len(chunks) == 0 or embeddings is None or len(embeddings) == 0:
        raise ValueError("Chunks or embeddings are missing or empty. Ensure a valid PDF is uploaded.")

    # Compute the query embedding
    query_embedding = model.encode([query])[0]
    # print(f"DEBUG: Query embedding: {query_embedding}")  # Debugging log

    # Compute cosine similarities
    similarities = np.array([cosine_similarity(query_embedding, emb) for emb in embeddings])
    # print(f"DEBUG: Chunk similarities: {similarities}")  # Debugging log

    # Find the most similar chunk
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]
    # print(f"DEBUG: Best Index: {best_idx}, Best Score: {best_score}")  # Debugging log

    # If the best score exceeds the threshold, return the most relevant sentence
    if best_score > threshold:
        relevant_chunk = chunks[best_idx]
        return get_most_relevant_sentence(relevant_chunk, query, model)
    else:
        return "Sorry, I didn’t understand your question. Do you want to connect with a live agent?"
