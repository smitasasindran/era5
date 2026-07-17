// Minimal BPE tokenizer engine that reads a real HuggingFace `tokenizers`
// tokenizer.json (normalizer + pre_tokenizer + BPE model) and re-implements
// encode() in plain JS. No external tokenizer library -- this mirrors the
// same word-frequency BPE merge algorithm used throughout the Python
// pipeline (see naive_bpe.py's encode_word), just ported to JS so it can
// run client-side against the *actual* trained tokenizer.json.

const SEP = ""; // separator unlikely to appear in real merge tokens

// Practical Unicode "word" class: letters + marks (so Devanagari/Telugu
// vowel signs stay attached to their base consonant) + digits + connector
// punctuation. Mirrors the HF `Whitespace` pre-tokenizer's `\w+|[^\w\s]+`
// (Unicode-aware \w), just spelled out with JS Unicode property escapes.
const WORD_RE = /[\p{L}\p{M}\p{N}\p{Pc}]+|[^\s\p{L}\p{M}\p{N}\p{Pc}]+/gu;

// This engine hand-implements exactly one pipeline: a BPE model, an optional
// Lowercase normalizer, and a Whitespace-style pre-tokenizer. It does NOT
// generically interpret tokenizer.json -- there's no code path for Unigram/
// WordPiece models, NFC/NFD/Sequence normalizers, or Metaspace/ByteLevel/
// Sequence pre-tokenizers. Retraining with a different vocab under the SAME
// pipeline is safe to drop in; changing the pipeline itself is not (see
// README's "Swapping in a new tokenizer.json" section) -- so we validate
// against an allowlist and fail loudly rather than silently mis-tokenizing.
const SUPPORTED_MODEL_TYPES = ["BPE"];
const SUPPORTED_NORMALIZER_TYPES = [null, "Lowercase"];
const SUPPORTED_PRE_TOKENIZER_TYPES = [null, "Whitespace"];

class UnsupportedTokenizerError extends Error {}

class BPETokenizer {
  constructor(json) {
    this.raw = json;
    this.normalizerType = json.normalizer?.type ?? null;
    this.preTokenizerType = json.pre_tokenizer?.type ?? null;
    this.modelType = json.model?.type ?? null;

    this.validate();

    const model = json.model;
    this.vocab = new Map(Object.entries(model.vocab));
    this.mergeRank = new Map();
    model.merges.forEach(([a, b], rank) => {
      this.mergeRank.set(a + SEP + b, rank);
    });
  }

  validate() {
    const problems = [];
    if (!SUPPORTED_MODEL_TYPES.includes(this.modelType)) {
      problems.push(
        `model.type is "${this.modelType}" -- this engine only implements BPE ` +
          `(merges + greedy lowest-rank pairwise merging). A Unigram or ` +
          `WordPiece model needs different encode logic entirely.`
      );
    }
    if (!SUPPORTED_NORMALIZER_TYPES.includes(this.normalizerType)) {
      problems.push(
        `normalizer.type is "${this.normalizerType}" -- this engine only ` +
          `implements Lowercase (or no normalizer). NFC/NFD/Strip/Sequence ` +
          `normalizers would be silently skipped instead of applied.`
      );
    }
    if (!SUPPORTED_PRE_TOKENIZER_TYPES.includes(this.preTokenizerType)) {
      problems.push(
        `pre_tokenizer.type is "${this.preTokenizerType}" -- this engine only ` +
          `implements a Whitespace-style regex. Metaspace/ByteLevel/Sequence ` +
          `pre-tokenizers would produce different, wrong atom boundaries.`
      );
    }
    if (problems.length > 0) {
      throw new UnsupportedTokenizerError(
        "This tokenizer.json uses a pipeline this playground doesn't support:\n" +
          problems.map((p) => `  - ${p}`).join("\n") +
          "\nUpdate tokenizer.js to handle it, or retrain under the same pipeline."
      );
    }
  }

  get vocabSize() {
    return this.vocab.size;
  }

  normalize(text) {
    if (this.normalizerType === "Lowercase") {
      return text.toLowerCase();
    }
    return text;
  }

  // Returns [{text, start, end}] offsets into the (already normalized) text.
  preTokenize(text) {
    const atoms = [];
    let m;
    WORD_RE.lastIndex = 0;
    while ((m = WORD_RE.exec(text)) !== null) {
      atoms.push({ text: m[0], start: m.index, end: m.index + m[0].length });
    }
    return atoms;
  }

  // BPE-merge a single pre-tokenized atom. Same greedy lowest-rank-first
  // algorithm as the Python reference (encode_word in naive_bpe.py).
  encodeAtom(word) {
    let symbols = Array.from(word); // iterate by codepoint, not UTF-16 unit
    if (symbols.length > 1) {
      while (true) {
        let bestRank = Infinity;
        let bestIdx = -1;
        for (let i = 0; i < symbols.length - 1; i++) {
          const rank = this.mergeRank.get(symbols[i] + SEP + symbols[i + 1]);
          if (rank !== undefined && rank < bestRank) {
            bestRank = rank;
            bestIdx = i;
          }
        }
        if (bestIdx === -1) break;
        symbols.splice(bestIdx, 2, symbols[bestIdx] + symbols[bestIdx + 1]);
      }
    }
    return symbols.map((piece) => ({
      piece,
      id: this.vocab.has(piece) ? this.vocab.get(piece) : null,
    }));
  }

  // Full tokenize: normalize -> pre-tokenize -> BPE per atom.
  // Returns { normalizedText, atoms, gaps } where atoms/gaps interleave in
  // original order so a caller can reconstruct the text visually.
  tokenize(text) {
    const normalized = this.normalize(text);
    const rawAtoms = this.preTokenize(normalized);

    const atoms = rawAtoms.map((a) => ({
      ...a,
      pieces: this.encodeAtom(a.text),
    }));

    // Gaps (whitespace/etc. the pre-tokenizer regex intentionally skips)
    // between/around atoms, so the UI can render them un-colored.
    const gaps = [];
    let cursor = 0;
    for (const a of atoms) {
      gaps.push(normalized.slice(cursor, a.start));
      cursor = a.end;
    }
    gaps.push(normalized.slice(cursor));

    return { normalizedText: normalized, atoms, gaps };
  }
}

async function loadTokenizer(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to load ${url}: ${res.status}`);
  const json = await res.json();
  return new BPETokenizer(json);
}

// Isomorphic export (no-op in the browser, where `module` is undefined) so
// this file can also be required from a Node test harness.
if (typeof module !== "undefined") {
  module.exports = { BPETokenizer, loadTokenizer, UnsupportedTokenizerError };
}
