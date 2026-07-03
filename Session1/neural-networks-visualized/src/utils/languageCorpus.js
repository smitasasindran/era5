// Tiny synthetic language used to demonstrate that word embeddings learn
// category structure purely from co-occurrence statistics, never from an
// explicit "these words are similar" label.

export const WORD_CATEGORIES = {
  animals: ["cat", "dog", "cow"],
  fruits: ["apple", "mango"],
  verbs: ["eat", "chase", "see"],
};

// Fixed order so every word has a stable vocabulary index.
export const VOCABULARY = Object.values(WORD_CATEGORIES).flat();

/**
 * Looks up which category a vocabulary word belongs to.
 * @param {string} word
 * @returns {string} one of the WORD_CATEGORIES keys, or "unknown".
 */
export function getCategoryOf(word) {
  for (const [category, words] of Object.entries(WORD_CATEGORIES)) {
    if (words.includes(word)) return category;
  }
  return "unknown";
}

function sample(list) {
  return list[Math.floor(Math.random() * list.length)];
}

/**
 * Generates `numSentences` animal → verb → fruit sentences (e.g.
 * "cat eat apple"). Consecutive sentences are meant to be concatenated
 * into one token stream (see `buildCorpus`): a sentence's fruit is
 * immediately followed by the next sentence's animal, so every word in
 * the vocabulary appears both as the "current" word and as the "next"
 * word somewhere in the corpus — required for every word's embedding to
 * actually receive a training signal.
 *
 * Every animal is always followed by a verb, every verb by a fruit, and
 * every fruit by the next sentence's animal. That per-category regularity
 * — not the specific word — is what gives same-category words identical
 * next-word statistics, which is exactly the signal embeddings latch onto.
 *
 * @param {number} [numSentences=60]
 * @returns {string[][]} an array of 3-word sentences.
 */
export function generateSentences(numSentences = 60) {
  const { animals, fruits, verbs } = WORD_CATEGORIES;
  return Array.from({ length: numSentences }, () => [sample(animals), sample(verbs), sample(fruits)]);
}

/**
 * Builds the full training corpus: the generated sentences, the flat token
 * stream they form when concatenated, and every consecutive (current,
 * next) word pair in that stream — the next-token-prediction training
 * signal.
 *
 * @param {number} [numSentences=60]
 * @returns {{sentences: string[][], tokenStream: string[], bigrams: Array<[string, string]>}}
 */
export function buildCorpus(numSentences = 60) {
  const sentences = generateSentences(numSentences);
  const tokenStream = sentences.flat();

  const bigrams = [];
  for (let i = 0; i < tokenStream.length - 1; i++) {
    bigrams.push([tokenStream[i], tokenStream[i + 1]]);
  }

  return { sentences, tokenStream, bigrams };
}

/**
 * Converts word bigrams into vocabulary-index bigrams for the embedding
 * model's tensors.
 *
 * @param {Array<[string, string]>} bigrams
 * @param {string[]} [vocabulary=VOCABULARY]
 * @returns {Array<[number, number]>}
 */
export function bigramsToIndices(bigrams, vocabulary = VOCABULARY) {
  return bigrams.map(([current, next]) => [vocabulary.indexOf(current), vocabulary.indexOf(next)]);
}

/**
 * Combines 2D-projected embedding coordinates with each word's category,
 * producing the {word, category, x, y} points the scatter plot expects.
 *
 * @param {number[][]} embeddings2D - One [x, y] pair per vocabulary word, in `vocabulary` order.
 * @param {string[]} [vocabulary=VOCABULARY]
 */
export function toEmbeddingPoints(embeddings2D, vocabulary = VOCABULARY) {
  return vocabulary.map((word, index) => ({
    word,
    category: getCategoryOf(word),
    x: embeddings2D[index][0],
    y: embeddings2D[index][1],
  }));
}
