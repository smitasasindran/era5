# Loss Harness — GPT Language Model

This notebook uses a small GPT-style model based on Karpathy's `build-nanogpt` implementation to make the language-model loss computation observable and verify its correctness.

## Configuration

```text
vocab_size = 50,257 (+ PAD and SEP tokens for later experiments)
n_layer    = 4
n_head     = 4
n_embd     = 256
```

---

## Experiment 1 — Tensor Shape Audit

The input tokens have shape `[B,T]`. The Transformer maps each token position to a `D`-dimensional hidden representation, producing `[B,T,D]`. The language-model head projects each hidden representation into the vocabulary space, producing `[B,T,V]` logits.

For next-token prediction, the final logit position has no corresponding target, so logits and targets are shifted to `[B,T-1,V]` and `[B,T-1]`. These are then flattened to `[B(T-1),V]` and `[B(T-1)]` for cross entropy.

Example:

```text
Input: "The cat sat on the mat"

tokens        [B,T]       = [1,6]
hidden        [B,T,D]     = [1,6,256]
logits        [B,T,V]     = [1,6,50257]

shift_logits  [B,T-1,V]   = [1,5,50257]
shift_targets [B,T-1]     = [1,5]

flat_logits   [T-1,V]     = [5,50257]
flat_targets  [T-1]       = [5]
```

There are 6 input tokens but only 5 next-token prediction targets because the final token has no target following it.

---

## Experiment 2 — Verify the Next-Token Shift

The language-model loss pairs each input position with the **next token** as its target.

```text
shift_inputs  = tokens[:, :-1]
shift_targets = tokens[:, 1:]
```

For example:

```text
Input token → Target token

"The"       → " cat"
" cat"      → " sat"
" sat"      → " on"
" on"       → " the"
" the"      → " mat"
```

The actual decoded token strings are printed rather than token IDs to visually verify that the shift is correct.

---

## Experiment 3 — Padding Mask

Padding positions must not contribute to the language-model loss.

A dedicated `<PAD>` token is introduced and a binary loss mask is aligned with the shifted targets.

For two sequences:

```text
Sequence 1: 3 valid tokens
Sequence 2: 6 valid tokens
```

there are:

```text
Valid input tokens:          9
Total prediction positions: 10
Contributing loss terms:     7
Masked/padding positions:    3
```

The distinction arises because each sequence loses its final position during the next-token shift.

The unmasked loss is:

```python
unmasked_loss = per_token_loss.mean()
```

The masked loss is:

```python
masked_loss = (
    per_token_loss * loss_mask
).sum() / loss_mask.sum()
```

Thus only valid target positions contribute to the final loss.

---

## Experiment 4 — Document Boundary Mask

Two independent documents are packed into one sequence:

```text
Document 1: A B C
Document 2: X Y Z

Packed:
A B C SEP X Y Z
```

After shifting:

```text
shift_inputs:   A   B   C  SEP  X   Y
shift_targets:  B   C  SEP  X    Y   Z
```

The prediction relationships are:

```text
A   → B
B   → C
C   → SEP
SEP → X       ← document boundary
X   → Y
Y   → Z
```

The `SEP → X` prediction crosses the document boundary and is therefore masked:

```python
boundary_mask[:, N1] = 0
```

The experiment compares the ordinary loss against the boundary-masked loss and verifies that exactly one prediction position is removed from the loss.

---

## Experiment 5 — Perplexity

Perplexity is related to cross entropy by:

$$
PPL = \exp(\text{CrossEntropy})
$$

For an approximately uniform predictor over `V` vocabulary tokens:

$$
CE \approx \ln V
$$

Therefore:

$$
PPL \approx V
$$

The randomly initialized model is evaluated before training. Its perplexity should therefore be in the neighborhood of the model's vocabulary size.

---

## Experiment 6 — Tied vs Untied LM Head

The model currently ties the input token embedding and output LM head:

```python
self.transformer.wte.weight = self.lm_head.weight
```

### Tied

The embedding and LM head share the same parameter matrix:

```text
Token embedding ──┐
                  ├── same weights
LM head ──────────┘
```

The shared matrix has:

$$
V \times D
$$

parameters.

### Untied

The embedding and LM head use separate matrices, requiring an additional:

$$
V \times D
$$

parameters.

The experiment compares the total parameter counts of tied and untied models and verifies that the tied model's embedding and LM-head weights share the same underlying storage.

For the current configuration:

```text
V = 50,259
D = 256

Parameters saved by tying:

50,259 × 256 = 12,866,304
```

---

## Experiment 7 — Ordinary vs Chunked Cross Entropy

The ordinary implementation materializes the complete `[N,V]`
logits tensor before computing cross entropy. The chunked
implementation computes the vocabulary projection and loss in
chunks, avoiding the full logits allocation.

With B=8, T=128, V=50,259 and chunk_size=32:

Ordinary CE peak GPU memory: 651.49 MB
Chunked CE peak GPU memory:  274.62 MB
Memory ratio:                2.37×

The two implementations produced identical loss values
(up to floating-point precision), while chunking reduced
peak GPU memory by approximately 58%.



