// Framework-agnostic vector similarity helpers, used to find nearest
// neighbors in embedding space.

/**
 * Cosine similarity between two equal-length vectors, in [-1, 1].
 */
export function cosineSimilarity(a, b) {
  const dotProduct = a.reduce((sum, value, i) => sum + value * b[i], 0);
  const normA = Math.sqrt(a.reduce((sum, value) => sum + value * value, 0));
  const normB = Math.sqrt(b.reduce((sum, value) => sum + value * value, 0));
  if (normA === 0 || normB === 0) return 0;
  return dotProduct / (normA * normB);
}

/**
 * Finds the `k` words whose embedding is most similar (by cosine
 * similarity) to the given word's embedding.
 *
 * @param {string} word
 * @param {string[]} vocabulary - Word order matching `embeddings`' rows.
 * @param {number[][]} embeddings - One row per vocabulary word.
 * @param {number} [k=3]
 * @returns {Array<{word: string, similarity: number}>}
 */
export function findNearestNeighbors(word, vocabulary, embeddings, k = 3) {
  const index = vocabulary.indexOf(word);
  if (index === -1) return [];

  const target = embeddings[index];
  return vocabulary
    .map((candidate, i) => ({ word: candidate, similarity: cosineSimilarity(target, embeddings[i]) }))
    .filter((entry) => entry.word !== word)
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, k);
}
