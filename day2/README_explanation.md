# 🧠 Day 2: The Brain (Loss, Backprop & Optimization)

> **Core Concept:** Day 1 built a network that *computes*. Day 2 makes it *improve*.
> Three questions: how wrong are we (loss), who is to blame (backprop), and how big
> a correction to make (optimizer).
>
> Every number in this file was produced by running the code in this folder.

---

## 0. The files

| File | Lines | What it builds | Run |
|---|---|---|---|
| `part1_loss_functions.py` | 775 | MSE, Cross-Entropy, softmax+CE, gradient checking | `python part1_loss_functions.py` |
| `part2_backpropagation.py` | 655 | The chain rule, layer-by-layer backward pass | `python part2_backpropagation.py` |
| `part3_training_loop.py` | 743 | The full loop + SGD / Momentum / Adam | `python part3_training_loop.py` |

---

## 1. The learning cycle

```text
[1. FORWARD PASS] ──► Network guesses: "Dog, 10% sure"
      │
      ▼
[2. LOSS]         ──► Truth = Dog (100%). Cross-Entropy.
      │               "Only 10% sure? ERROR = 2.3"
      ▼
[3. BACKPROP]     ──► The chain rule, travelling backwards.
      │               "Weight #45 caused 20% of this. Weight #46 caused 2%."
      ▼
[4. OPTIMIZE]     ──► W = W − (learning_rate × gradient)
                      The network is now slightly smarter.
```

### The five-step algorithm — all of deep learning

```text
1. FORWARD    input → layers → prediction        (Day 1)
2. LOSS       prediction vs target → one number  (part 1)
3. BACKWARD   loss → gradient for every weight   (part 2)
4. UPDATE     weight -= lr × gradient            (part 3)
5. REPEAT     until the loss stops falling

That is the entire field. Everything else optimizes these five lines.
```

---

## 2. Loss — how wrong are we?

```text
MSE:            (1/n) Σ (pred − target)²
Cross-Entropy:  −Σ target · log(pred)
```

Both are zero when perfect. So why is only one used for classification?

### ⭐ The reason, in real numbers

From `part1`, when the network is **confidently wrong** (true class at p = 0.01):

```text
  MSE gradient:  −0.66     "hey, maybe adjust a bit?"
  CE  gradient:  −100.0    "EMERGENCY! FIX THIS NOW!"
```

When it's nearly right:

```text
  MSE gradient:  −0.007    tiny nudge
  CE  gradient:  −1.01     tiny nudge
```

**That asymmetry is the whole argument.** Cross-Entropy's gradient is `−1/p`, so as
`p → 0` it explodes. Being confidently wrong is punished enormously. MSE's gradient
stays small, so a badly wrong network barely corrects itself and can stall.

> **Intuition:** MSE treats a 1% prediction and a 40% prediction as similarly bad.
> Cross-Entropy treats 1% as a catastrophe. For classification, it *is* one.

### The beautiful gradient

Why softmax and cross-entropy are always paired:

```text
∂L/∂logits = predicted − target
```

That's the whole thing. The `exp` in softmax and the `log` in cross-entropy cancel
exactly.

```text
predicted = [0.7, 0.2, 0.1]
target    = [1.0, 0.0, 0.0]
gradient  = [−0.3, 0.2, 0.1]     ← "raise the first, lower the others"
```

**The gradient literally *is* the error.** Cheapest correct gradient in machine
learning — which is why every classifier uses this pair.

---

## 3. Backpropagation — who is to blame?

The **chain rule**, applied backwards:

```text
dz/dx = dz/dy · dy/dx
```

Think of a factory line:

```text
Raw Material (X) ──► Machine A ──► Machine B ──► Defective Product
```

Machine B says *"I erred, but I got bad parts from A"*, computes its local gradient,
and passes the blame backward. A adjusts its settings by the blame it received.

| Layer | Backward pass |
|---|---|
| Dense | `dW = inputᵀ · d_out`, `db = d_out`, `dx = d_out · Wᵀ` |
| ReLU | `dx = d_out × (input > 0)` — a gate: pass or block |
| Softmax + CE | `d_logits = probs − target` |

**ReLU's backward pass deserves a pause.** It's a switch: positive input → gradient
passes untouched; negative → gradient killed. That is precisely *why* dead ReLU
units never recover — no gradient ever reaches them again.

### ⭐ Gradient checking — the technique that saves you

Verify any hand-derived gradient against a numerical one:

```text
numerical ≈ [f(x + h) − f(x − h)] / 2h        with h = 1e-5
```

`part2` does this for every layer:

```text
  Dense2 weights max error: 0.00000044 ✓ PASS
  Dense1 weights max error: 0.00000008 ✓ PASS
```

~1e-7. Anything below ~1e-5 means your derivation is right.

> **The single most useful debugging technique in deep learning.** A wrong gradient
> doesn't crash — the network just trains badly and you spend a week blaming the
> learning rate.
>
> It's also the same idea as Day 7's golden model: **check the implementation
> against an independent reference** instead of trusting that it looks right.

---

## 4. Training — the loop that learns

Real output from `part3` (3 classes, 300 samples, 1000 epochs, lr = 0.05):

```text
  Accuracy BEFORE training: 3.3% (10/300)   [random guessing = 33.3%]
  During epoch 0:  loss = 0.1422, accuracy = 96.7%
  End:             loss = 0.0002, accuracy = 100.0%
```

**Those three lines contain a trap.**

`3.3%` is *worse than random* (33.3%). Not a bug — a randomly initialized network
isn't neutral, it's **systematically** wrong, confidently predicting nonsense until
gradients correct it.

`96.7%` during epoch 0 looks like a good start. It isn't. By the end of epoch 0 the
network had already taken **300 gradient steps** (one per sample), so that figure
averages accuracy *while learning*. Nearly all learning happens in the first epoch.

> ⚠️ This file used to print 96.7% labelled "random guessing" — wrong twice over.
> The real story, **3.3% → 100%**, is far more impressive than the original claim.
> Fixed; see §7.

---

## 5. Gradient descent & the learning rate

Blindfolded on a mountainside, trying to reach the valley:

```text
       * (high error)
        \
         \    step = lr × gradient
          *───►
           \
            * (minimum)
```

* **Gradient** — the slope under your feet; which way is up
* **Update** — you want *down*, so subtract it
* **Learning rate** — how big a step

Measured over 500 epochs:

| lr | Final loss | Verdict |
|---|---|---|
| 0.001 | 0.0100 | too slow — barely learns |
| 0.010 | 0.0025 | slow but converging |
| **0.050** | **0.0004** | **best** |
| 0.500 | 0.0076 | too high — overshoots |

**0.5 is worse than 0.01.** Bigger is not faster past a point — you step *over* the
valley instead of into it. Remember this table; it explains §6.

---

## 6. Optimizers — SGD, Momentum, Adam

| Optimizer | Update rule | Final loss (500 epochs) |
|---|---|---|
| **SGD** | `W -= lr · grad` | 0.0004 |
| **Momentum** | `v = βv − lr·grad; W += v` | **0.0002** |
| **Adam** | per-weight adaptive lr | **0.0000** |

**Momentum** keeps a running velocity, so consistent directions build speed and
noisy ones cancel — a ball rolling downhill rather than re-deciding every step.

**Adam** = Momentum + a *per-weight* learning rate scaled by recent gradient
magnitude. Rarely-updated weights get bigger steps. This is why `torch.optim.Adam`
is the default almost everywhere.

### ⚠️ The subtlety that made momentum look bad

This file originally used `lr = 0.05` for both SGD and Momentum, and reported
momentum as **worse** (0.0266 vs 0.0004). That looked like evidence against
momentum. It was an unfair comparison.

**Momentum multiplies your effective learning rate.** With β = 0.9 the velocity
converges to about `1/(1−β) = 10×` a single step:

```text
effective_lr ≈ lr / (1 − β)

  lr = 0.05, β = 0.9   →   effective ≈ 0.50
```

And §5 already measured that **lr = 0.5 is too high**. The original comparison
wasn't testing momentum — it was re-testing an overshooting learning rate.

Measured, everything else fixed, 500 epochs:

| Momentum lr | Effective lr | Final loss | |
|---|---|---|---|
| 0.05 | ~0.50 | 0.0266 | the unfair comparison |
| **0.01** | ~0.10 | **0.0002** | **beats SGD** |
| 0.005 | ~0.05 | 0.0004 | exactly ties SGD |

> **⭐ Rule of thumb: when you add momentum β, multiply your learning rate by (1−β).**

That last row matters: momentum at the *equivalent* effective rate gives **identical**
loss to SGD. Not a coincidence — it's proof the implementation is correct.

---

## 7. 🐛 Two bugs found and fixed here

**1. The baseline accuracy was misreported.** `part3` printed `acc_history[0]` and
called it "random guessing", but that's the average over epoch 0 — *after* 300
weight updates. Fixed by measuring before the first update: **3.3%**, not 96.7%.
`daily_log.md` corrected too.

**2. The momentum comparison was 10× unfair.** Momentum used `lr = 0.05` with
β = 0.9 (effective ~0.50) against SGD at 0.05. Changed to `lr = 0.01` with the
`(1−β)` rule documented in the code. Momentum now correctly beats SGD, and the
ordering teaches the intended lesson: **Adam < Momentum < SGD**.

---

## 8. Where to use it & why

* **Phase:** training only — on a GPU or workstation, once.
* **Hardware:** massive RAM and matrix throughput.
* **Never on the drone.** Backprop must store *every* intermediate activation from
  the forward pass, and gradients are as large as the weights — roughly 3× the
  memory of inference.

> Train once on a big machine. Infer forever on a small one. You run this until the
> network is smart, freeze the weights, and ship only Day 1's forward pass —
> quantized to int8 by Day 5.

---

## ✅ Self-test

1. Why Cross-Entropy over MSE for classification? Give the numbers.
2. What is `∂L/∂logits` for softmax + CE, and why is it so simple?
3. Write ReLU's backward pass. Why do dead ReLU units never recover?
4. How do you verify a gradient you derived by hand?
5. Your untrained 3-class net scores 3.3%. Random is 33%. Is it broken?
6. Why is lr = 0.5 worse than lr = 0.05?
7. You add momentum β = 0.9 and training gets worse. What did you forget?

<details><summary>Answers</summary>

1. CE's gradient is `−1/p`, so confident mistakes are punished enormously: at p=0.01, CE gives −100.0 vs MSE's −0.66. MSE barely corrects a badly wrong network.
2. `predicted − target`. The `exp` in softmax and `log` in CE cancel exactly, so the gradient *is* the error.
3. `dx = d_out × (input > 0)`. A gate. Once a unit's input is always negative, no gradient flows back, so its weights never change again.
4. Numerical gradient checking: `[f(x+h) − f(x−h)] / 2h`, h = 1e-5. Below ~1e-5 error means correct; this file reaches ~1e-7.
5. No. Random init isn't neutral — the net is confidently wrong in a consistent direction until gradients fix it. Below-chance at init is normal.
6. It overshoots, stepping over the valley rather than into it. Loss oscillates instead of settling.
7. To scale the learning rate down. Momentum multiplies effective lr by ~1/(1−β) = 10×. Multiply lr by (1−β).

</details>

---

**Next:** `day3/` — from 3 toy classes to real MNIST digits, where data
normalization stops being optional.
