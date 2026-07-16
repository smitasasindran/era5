"""
Naive (but reasonably efficient) word-frequency BPE, following Sennrich et al. (2015).

Words are pre-tokenized by whitespace splitting (naive pretokenizer).
Each word is represented as a tuple of unicode characters + an end-of-word marker.
We repeatedly merge the most frequent adjacent symbol pair (weighted by word
frequency) until the target vocab size is reached. Merges are tracked with an
incremental pair-frequency index so we don't rescan the whole corpus every step.
"""
from collections import defaultdict, Counter

EOW = "</w>"


def word_freqs_from_text(text):
    """Naive pretokenization: split on whitespace."""
    return Counter(text.split())


def word_to_symbols(word):
    return tuple(list(word) + [EOW])


class BPETokenizer:
    def __init__(self):
        self.merges = []          # ordered list of (a, b) -> merged token, in learned order
        self.merge_rank = {}       # (a, b) -> rank (lower = learned earlier = applied first)
        self.vocab = set()         # all symbols (base chars + merged tokens)

    def train(self, combined_word_freqs, vocab_size, verbose=True):
        # word representation: dict[tuple_of_symbols] = freq
        words = {word_to_symbols(w): f for w, f in combined_word_freqs.items()}

        base_symbols = set()
        for w in words:
            base_symbols.update(w)
        self.vocab = set(base_symbols)

        if verbose:
            print(f"Base vocab size (unique symbols): {len(self.vocab)}")

        num_merges = vocab_size - len(self.vocab)
        if num_merges <= 0:
            print("vocab_size <= base vocab size; no merges to learn.")
            return

        # pair_freq[(a,b)] = total frequency across all words
        # pair_to_words[(a,b)] = set of word-tuples currently containing that pair
        pair_freq = defaultdict(int)
        pair_to_words = defaultdict(set)

        def add_word_pairs(w, freq):
            for i in range(len(w) - 1):
                pair = (w[i], w[i + 1])
                pair_freq[pair] += freq
                pair_to_words[pair].add(w)

        for w, f in words.items():
            add_word_pairs(w, f)

        for step in range(num_merges):
            if not pair_freq:
                if verbose:
                    print(f"No more pairs to merge at step {step}, stopping early.")
                break

            best_pair = max(pair_freq.items(), key=lambda kv: kv[1])[0]
            best_freq = pair_freq[best_pair]
            if best_freq < 2:
                if verbose:
                    print(f"Best remaining pair freq < 2 at step {step}, stopping early.")
                break

            merged_token = best_pair[0] + best_pair[1]
            self.merges.append(best_pair)
            self.merge_rank[best_pair] = len(self.merges) - 1
            self.vocab.add(merged_token)

            affected_words = list(pair_to_words[best_pair])
            for old_w in affected_words:
                freq = words.pop(old_w, None)
                if freq is None:
                    continue

                # remove old pair contributions for this word
                for i in range(len(old_w) - 1):
                    p = (old_w[i], old_w[i + 1])
                    pair_freq[p] -= freq
                    if pair_freq[p] <= 0:
                        del pair_freq[p]
                    pair_to_words[p].discard(old_w)

                # build new word with merges applied (just this one merge, greedily left-to-right)
                new_w = []
                i = 0
                while i < len(old_w):
                    if i < len(old_w) - 1 and (old_w[i], old_w[i + 1]) == best_pair:
                        new_w.append(merged_token)
                        i += 2
                    else:
                        new_w.append(old_w[i])
                        i += 1
                new_w = tuple(new_w)

                words[new_w] = words.get(new_w, 0) + freq
                add_word_pairs(new_w, freq)

            # best_pair should now be gone
            pair_freq.pop(best_pair, None)
            pair_to_words.pop(best_pair, None)

            if verbose and (step + 1) % 500 == 0:
                print(f"  merge {step + 1}/{num_merges}: {best_pair} -> {merged_token} (freq {best_freq}), vocab={len(self.vocab)}")

        if verbose:
            print(f"Final vocab size: {len(self.vocab)} ({len(self.merges)} merges learned)")

    def encode_word(self, word):
        symbols = list(word_to_symbols(word))
        if len(symbols) == 1:
            return symbols

        while True:
            # find the lowest-rank (earliest learned) mergeable adjacent pair
            best_rank = None
            best_i = None
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                rank = self.merge_rank.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_i = i
            if best_i is None:
                break
            symbols[best_i:best_i + 2] = [symbols[best_i] + symbols[best_i + 1]]
        return symbols

    def encode_text(self, text):
        tokens = []
        for word in text.split():
            tokens.extend(self.encode_word(word))
        return tokens
