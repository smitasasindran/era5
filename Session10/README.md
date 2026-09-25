# Training Loop Analysis


## Assignment

Take a small model and a real loop, and make it tell you the truth about itself.
1. **Print every tensor shape in the step**, and write one line saying what each dimension means.
2. **Verify one gradient by hand.** Nudge a weight, measure how the loss changed, and compare against what `backward()` reported.
3. **Break gradient accumulation on purpose.** Use the average of the averages with micro-batches of different lengths, and plot both curves together so the gap is visible.
4. **Log the grad norm at every step**, then find one step where it moved before the loss did.
5. **Compute your own MFU**, report it honestly, and explain what is costing you the distance to 40%.
6. **Represent `0.1` in fp32, bf16, and fp8 E4M3**, showing the bits, then explain which format you would train in and why.

---

# 1. Training Shapes

The model is based on Andrej Karpathy's [nanoGPT](https://github.com/karpathy/build-nanogpt).
Implementation details are in the Colab notebook.

## Model configuration

```text
B = 8       # Batch size
T = 128     # Sequence length

vocab_size = 50257
n_layer    = 4
n_head     = 4
n_embd     = 256
```

## Shape analysis

```text
Input:
tokens                                  (8, 128)             # (B, T)
embeddings (tok+pos)                    (8, 128, 256)        # (B, T, C)

CausalSelfAttention:
 q projected                            (8, 128, 256)        # (B, T, C)
 k projected                            (8, 128, 256)        # (B, T, C)
 v projected                            (8, 128, 256)        # (B, T, C)

 q after head split                     (8, 4, 128, 64)      # (B, H, T, D)
 k after head split                     (8, 4, 128, 64)      # (B, H, T, D)
 v after head split                     (8, 4, 128, 64)      # (B, H, T, D)

scaled_dot_product_attention:
 attention scores*                      (8, 4, 128, 128)*    # (B, H, T_query, T_key)
 attention weights*                     (8, 4, 128, 128)*    # (B, H, T_query, T_key)
 output                                 (8, 4, 128, 64)      # (B, H, T, D)

 * implicit/intermediate shapes; not returned by
   F.scaled_dot_product_attention()

output attn (merged heads)              (8, 128, 256)        # (B, T, C)
output projection                       (8, 128, 256)        # (B, T, C)

MLP:
 input                                  (8, 128, 256)        # (B, T, C)
 hidden                                 (8, 128, 1024)       # (B, T, 4C)
 output                                 (8, 128, 256)        # (B, T, C)

transformer output                      (8, 128, 256)        # (B, T, C)
LM head logits                          (8, 128, 50257)      # (B, T, V)

Loss:
 logits used                            (8, 127, 50257)      # (B, T-1, V)
 targets                                (8, 127)             # (B, T-1)
 logits_flat                            (1016, 50257)        # (B*(T-1), V)
 targets_flat                           (1016,)              # (B*(T-1),)
 loss                                   ()                   # scalar
```

### Final loss

```text
t+1 loss: 10.344536
```

### Dimension key

| Symbol | Meaning | Value |
|---|---|---:|
| `B` | Batch size | 8 |
| `T` | Sequence length | 128 |
| `C` | Model / embedding dimension | 256 |
| `H` | Number of attention heads | 4 |
| `D` | Dimension per attention head | 64 |
| `V` | Vocabulary size | 50,257 |

### Sanity checks

```text
C = H × D = 4 × 64 = 256
B × (T-1) = 8 × 127 = 1016
```

---

# 2. Verify One Gradient by Hand

For the gradient check, pick a random parameter, eg attention projection weight from the first transformer block:

```python
param = model.transformer.h[0].attn.c_attn.weight
```

Selected original value:

```text
0.0196425449103117
```

## 2.1 Compute the autograd gradient

Run `backward()` and record PyTorch's gradient $\( \frac{\partial L}{\partial w} \)$ for the selected weight parameter  

```text
loss: 10.829090118408203
autograd gradient: -7.822928455425426e-05
```

## 2.2 Manually change the weight and observe the loss

Select a small perturbation, for example:

```text
ε = 1e-2
```

Change the selected parameter:

```python
w = w + epsilon
```

Then compute the loss again using the **same tokens**.

The finite-difference estimate is:  

$$
\[
\frac{L(w+\epsilon)-L(w)}{\epsilon}
\]
$$


## 2.3 Compare the gradients

The two gradients come from independent mechanisms:

```text
autograd_grad:
    computational graph
           ↓
       backward()
           ↓
          dL/dw


numerical_grad:
    perturb w → w+ε
           ↓
       measure Δloss
           ↓
          ΔL/ε
```


For `ε = 0.02`:

```text
Original weight w:          0.0196425449
Epsilon:                    2.0e-02

Loss(w):                    10.82909011840820312500
Loss(w + epsilon):          10.82908916473388671875
Δloss:                      -9.53674316406250000000e-07

Autograd gradient (∂L/∂w):  -0.000078229277278
Numerical gradient (ΔL/​ϵ):  -0.000047683715820
Absolute error:             3.054556145798415e-05
Relative error:             3.904620178110540e-01
```

The Autograd gradient and Numerical gradient have a relative difference of 39%  

## 2.4 Understanding the difference

The gradient reported by PyTorch is:  

$$
\[
\frac{\partial L}{\partial w} = -7.8229\times10^{-5}
\]
$$

This is the derivative of the loss with respect to the selected weight at its original value. In other words, it describes the instantaneous rate at which the loss changes for a very small change in the weight.

### Predicting the change in loss

For a sufficiently small perturbation, the first-order approximation is:    

$$
\[
\Delta L \approx
\frac{\partial L}{\partial w}\epsilon
\]
$$    

With `ε = 0.02` and `(∂L/∂w) = -0.000078229277278`:

$$
\[
\Delta L
\approx
-7.8229\times10^{-5}\times0.02
\approx -1.56458\times10^{-6}
\]
$$

This is the **expected** change in loss based on the autograd gradient.


The actual observed change was:

$$  
\[\Delta L
=L(w+\epsilon)-L(w)
=-9.5367431640625\times10^{-7}
\]
$$


The observed change has the same sign as the predicted change: both are negative. Increasing this weight decreases the loss, which is consistent with the negative autograd gradient. However, the magnitudes differ.

This difference is expected because the first-order approximation assumes that the loss is approximately linear over the perturbation interval. Here, `ε = 0.02` is not infinitesimally small, so curvature in the loss function can cause the actual change to differ from the linear prediction.

### Numerical gradient

The forward finite-difference estimate is:

$$
\[
\text{Numerical gradient}
=\frac{\Delta L}{\epsilon}
\]
$$

Therefore:

```text
Numerical gradient
= -9.536743164062500e-07 / 0.02
= -4.76837158203125e-05
```

Comparison:

```text
Autograd gradient:  -7.8229277278e-05
Numerical gradient: -4.7683715820e-05
```

The finite-difference estimate has approximately **39% relative error**.

### Why isn't the numerical gradient exactly the same?

The numerical gradient is an approximation to the derivative. With a forward difference,

$$
\[
\frac{L(w+\epsilon)-L(w)}{\epsilon}
\]
$$

we are measuring the average slope of the loss between $\(w\)$ and $\(w+\epsilon\)$, rather than the exact instantaneous slope at $\(w\)$.

If the loss were perfectly linear over this interval, the two would be identical. In practice, the loss surface has curvature, so the slope changes as the weight moves from $\(w\)$ to $\(w+\epsilon\)$.

A larger $\(\epsilon\)$ therefore introduces more finite-difference approximation error. Reducing $\(\epsilon\)$ should generally make the numerical gradient approach the autograd gradient until floating-point precision starts to dominate.

---

# 3. Break Gradient Accumulation on Purpose

The goal is to deliberately implement the **average of averages** with micro-batches of different lengths, then compare it with correct token-weighted gradient accumulation.

## 3.1 Why average-of-averages is wrong

Suppose we have two micro-batches:

```text
Micro-batch 1: 100 tokens
Micro-batch 2: 20 tokens
```

and their average losses are:

```text
batch 1 loss = 2.0
batch 2 loss = 4.0
```

The naive average of the two micro-batch losses is:

$$
\[
\frac{2.0+4.0}{2}=3.0
\]
$$

But if every token should have equal importance, the correct combined loss is:

$$
\[
\frac{100\times2.0+20\times4.0}{100+20}
=2.333\ldots
\]
$$

The first approach gives the 20-token micro-batch the same weight as the 100-token micro-batch:

```text
average of averages = 3.0
token-weighted average = 2.333...
```

The second approach is the correct one.

## 3.2 Deterministic training batches with different token sizes

### Test setup

Create two models:

- one using correct gradient accumulation
- one using intentionally incorrect accumulation

Both models start with exactly the same weights.  
For each optimizer step, generate deterministic micro-batches. Each micro-batch can have a different sequence length `T ∈ [32, 128]`.   
The same micro-batches are passed to both models so that the comparison isolates the accumulation strategy.   
A fixed evaluation batch is generated once and reused throughout the experiment.   

## 3.3 Training loop

For each optimizer step:

1. Create the same micro-batches for both models.
2. Count the total number of prediction tokens across **all** micro-batches in the accumulation window.
3. Accumulate gradients across the configured number of micro-batches.
4. Apply one optimizer update.
5. Evaluate both models on exactly the same fixed data.

### Correct gradient accumulation

Each micro-batch is weighted according to the number of prediction tokens it contains:

```python
weight = n_tokens / total_tokens
(loss_correct * weight).backward()
```

### Intentionally broken accumulation

Every micro-batch is given equal weight:

```python
(loss_broken / grad_accum_steps).backward()
```

This implements an **average of micro-batch averages**, rather than a token-weighted average.

> **Graph placeholder:** Add "Token-weighted loss vs Average of averages" graph here.

## 3.4 Analysis

The plot compares the loss curves from the two training runs.

The models start from identical weights and see identical micro-batches, but use different weighting when accumulating gradients. One uses token-weighted loss, while the other uses average-of-averages for gradient accumulation.

The losses initially track closely, then gradually diverge. After  ~2,000 optimizer steps, the token-weighted run has a lower loss on the fixed diagnostic batch than the average-of-averages run.

This continuues upto ~9,000 optimizer steps, after which the losses start converging.

<img width="686" height="470" alt="Grad Acc Loss Curves" src="https://github.com/user-attachments/assets/78405d15-86ad-43d9-af25-6f9eec067680" />

---

# 4. Gradient Norm Analysis

The fourth experiment logs the gradient norm at every optimizer step and investigates whether the gradient signal can move before the loss visibly responds.

The gradient norm is measured **after all micro-batch `backward()` calls and before `optimizer.step()`**. The loss is measured on the fixed diagnostic batch after the optimizer update.

Because the raw gradient norm is noisy, the analysis uses a smoothed gradient signal to identify candidate regions. The raw gradient values are then inspected around the selected candidate steps.

The initial transient portion of training (1000 steps) is excluded from candidate selection because both the gradient norm and loss change substantially during this phase. The goal is to identify a later training step where the gradient signal changes while the loss remains relatively unchanged, followed by a subsequent loss response.

**Smoothed Gradient-norm/loss plot**
<img width="1189" height="790" alt="Grad Norm and Loss - smooth" src="https://github.com/user-attachments/assets/3f29d620-50cb-4a11-b974-cd2a3a8c5371" />


## 4.1 Gradnorm / Loss plots 

Things to note: 
- A larger gradient norm does not necessarily mean loss will increase or decrease;
- A smaller gradient norm does not necessarily mean loss will move in a particular direction;
- What matters for the loss change is the direction of the update, optimizer state, curvature, and the data being evaluated;

<img width="998" height="690" alt="gradnorm-8122" src="https://github.com/user-attachments/assets/2447209f-1796-4c86-8ca1-110b2916aead" />


> **Graph placeholder:** Add more candidate-step local plots here.

---

# 5. Compute MFU (Model FLOPS Utilization)

MFU is:

$$
\[
\text{MFU}
=\frac{\text{FLOPs the training actually performs per second}}
{\text{GPU's theoretical peak FLOPs per second}}
\]
$$

There are three things to determine:

1. Estimate FLOPs per training token for the model.
2. Measure tokens/second during actual training.
3. Determine the GPU's theoretical peak FLOPs for the precision being used.

## 5.1 Calculate FLOPs per token

For a GPT-style transformer, a common approximation is:

$$
\[
\text{FLOPs/token}\approx6N
\]
$$

where $\(N\)$ is the number of non-embedding model parameters.

The `6N` approximation comes from:

- forward pass: approximately `2N` FLOPs
- backward pass: approximately `4N` FLOPs

Therefore:

$$
\[
2N+4N\approx6N
\]
$$

There is also an additional FLOP contribution from self-attention that depends on sequence length.

Karpathy's nanoGPT MFU calculation uses approximately:

$$
\[
\text{FLOPs/token}
=6N+12Lhd_hT
\]
$$

where:

- $\(N\)$ = non-embedding parameters
- $\(L\)$ = number of transformer layers
- $\(h\)$ = number of attention heads
- $\(d_h\)$ = dimension of each attention head
- $\(T\)$ = sequence length

Since: $\[hd_h=C\]$, 

this can also be written as:

$$
\[
\text{FLOPs/token}
=6N+12LCT
\]
$$

**Measured Values**:
```text
Total parameters:       16,058,112
Embedding parameters:   12,898,560
Non-embedding params:   3,159,552
FLOPs/token:            20,530,176
```

## 5.2 Measure tokens/second

Measure the actual training throughput in tokens/second during the training loop.

Because the experiment uses variable sequence lengths, the number of processed tokens is calculated from the actual micro-batches rather than assuming a fixed `T`.

**Measured Value**:
```text
FLOPs/token:     20,530,176
Tokens/sec:      21,027
Achieved FLOPs:  431,681,520,854
Achieved TFLOPs: 0.4316815208535321```


## 5.3 GPU peak FLOPs

The Colab runtime used a **NVIDIA Tesla T4**.

NVIDIA specifies:

- **8.1 TFLOPS FP32**
- **65 TFLOPS FP16 Tensor Core performance**

For this experiment, the model weights are FP32 and autocast was configured for FP16 but was not actually active. Therefore, the training arithmetic was FP32 and the FP32 peak is used for the MFU calculation.

## 5.4 MFU result

| Parameter | Result |
|---|---:|
| GPU | Tesla T4 |
| Model weights | FP32 |
| Autocast configured dtype | FP16 |
| Autocast actually active | No |
| Training arithmetic | FP32 |
| T4 FP32 peak | 8.1 TFLOPS |
| Training throughput | 21,260 tokens/s |
| FLOPs/token | 20,530,176 |
| Measured FLOPs/s | 436,476,011,710 |
| Measured TFLOPs/s | **0.436 TFLOPS/s** |

Therefore:

$$
\[
P_{\text{peak}}=8.1\text{ TFLOP/s}
\]
$$

and:

$$
\[
\text{MFU}
=
\frac{0.4365}{8.1}
\approx0.0539
\]
$$

Therefore:

$$
\[
\boxed{\text{MFU}\approx5.4\%}
\]
$$

### Why is MFU far below 40%?

Several aspects of this experiment work against high GPU utilization.

#### 1. Very small model

The transformer is small and therefore does not provide enough work to keep the GPU fully occupied.

#### 2. Small batch / micro-batch size

The experiment uses:

```text
B = 4
grad_accum_steps = 4
```

Gradient accumulation increases the effective batch size, but each individual micro-batch is still small. The setup was designed for the gradient-accumulation experiment rather than maximum GPU utilization.

#### 3. Variable sequence lengths

The experiment deliberately generates:

```text
T = 32 ... 128
```

This is useful for demonstrating the gradient-accumulation bug, but irregular sequence lengths are not ideal for GPU efficiency.

#### 4. CPU → GPU transfers

`get_batch()` performs:

```python
return x.to(device)
```

for every micro-batch.

Thus the loop repeatedly performs:

```text
CPU tensor
    ↓
GPU transfer
    ↓
forward
    ↓
backward
```

This can contribute significantly to the gap, particularly with a small model.

#### 5. Python and framework overhead

For a tiny model, Python-level and framework overhead can become a significant fraction of total runtime.

> **Graph/table placeholder:** Add MFU calculation and/or throughput visualization here.

---

# 6. Number Formats: FP32, BF16, and FP8 E4M3

The assignment asks us to take the number `0.1`, represent it in:

- FP32
- BF16
- FP8 E4M3

and show the bits. Then we choose a training format and explain why.

## 6.1 Binary representation of 0.1

Decimal numbers such as: $\[0.123\] represent powers of 10:

$$
\[
0\times10^{-1}
+
1\times10^{-2}
+
2\times10^{-3}
\]
$$

Binary works the same way, except the powers are powers of 2.

### Binary to decimal

For example: $\[0.101_2\]$ means:

$$
\[
1\times2^{-1}
+
0\times2^{-2}
+
1\times2^{-3}
\]
$$

which is:

$$
\[
\frac12+0+\frac18
=0.5+0.125
=0.625
\]
$$

Therefore:

```text
0.101₂ = 0.625₁₀
```

### Decimal to binary

We want to find:

$$
\[
0.1_{10}=?_2
\]
$$

To convert a fraction from decimal to binary, repeatedly multiply the fractional part by 2. The integer part becomes the next binary digit.

| Step | Fraction × 2 | Binary digit | New fraction |
|---:|---:|:---:|---:|
| 1 | `0.1 × 2 = 0.2` | 0 | 0.2 |
| 2 | `0.2 × 2 = 0.4` | 0 | 0.4 |
| 3 | `0.4 × 2 = 0.8` | 0 | 0.8 |
| 4 | `0.8 × 2 = 1.6` | 1 | 0.6 |
| 5 | `0.6 × 2 = 1.2` | 1 | 0.2 |
| 6 | `0.2 × 2 = 0.4` | 0 | 0.4 |
| 7 | `0.4 × 2 = 0.8` | 0 | 0.8 |
| 8 | `0.8 × 2 = 1.6` | 1 | 0.6 |
| 9 | `0.6 × 2 = 1.2` | 1 | 0.2 |

Collecting the binary digits:

```text
000110011001100110011...
```

Therefore:

$$
\[
0.1_{10}
=0.000110011001100110011\ldots_2
\]
$$

### Why doesn't the representation terminate?

Some decimal fractions have finite binary representations, but `0.1` does not.

A number can be represented exactly as a finite binary fraction only if its denominator contains no prime factors other than 2.

But:

$$
\[
0.1=\frac{1}{10}
\]
$$

and 10 contains a factor of 5. Binary fractions can only represent finite sums of powers of \(1/2\), so \(1/10\) cannot be represented exactly.

Thus the binary representation repeats:

```text
0.000110011001100110011...
```

This is analogous to:

```text
1/3 = 0.333333...
```

> **Graph/table placeholder:** Add FP32, BF16, and FP8 E4M3 bit representations here.
