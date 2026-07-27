# 🏗️ Day 3: The Data Engineer (Scaling to Reality)

> **Core Concept:** Days 1–2 built a math engine on 3 toy images. Day 3 feeds it
> real handwritten digits — and real data will break your network if you mishandle it.
>
> Every number in this file was produced by running the code in this folder.

---

## 0. The file

| File | Lines | What it builds | Run |
|---|---|---|---|
| `part1_mnist_classifier.py` | 667 | IDX parsing, normalization, a 3-layer net, full training | `python part1_mnist_classifier.py` |

**Data is not in the repo** (11.5 MB of `.gz`). The script's `download_mnist()`
fetches it into `mnist_data/` on first run.

---

## 1. The data pipeline

```text
[Raw IDX files]    ──► binary, big-endian, custom format from 1998
      │                 not CSV, not JSON — you parse bytes by hand
      ▼
[Memory loading]   ──► shape (60000, 28, 28), dtype uint8
      │
      ▼
[Flattening]       ──► a Dense layer wants a 1D vector, not a 2D grid
      │                 28 × 28 = 784  →  shape (60000, 784)
      ▼
[Normalization]    ──► ⚠️ CRITICAL. Raw pixels are 0–255.
      │                 pixel / 255.0  →  range [0.0, 1.0]
      ▼
[One-hot encoding] ──► the network can't read "7"
      │                 label 7  →  [0,0,0,0,0,0,0,1,0,0]
      ▼
[Shuffle + train]  ──► feed to the Day 1 forward pass and Day 2 backprop
```

### Why normalization is not optional

With raw pixels at 255 and a 784-input layer, a single neuron's pre-activation is a
sum of 784 terms each up to `255 × w`. The logits explode, `exp()` in softmax
overflows, and gradients become `inf` or `NaN` on the first step.

Dividing by 255 caps every input at 1.0, so the sum stays in a range where float32
behaves. **One line of code decides whether the network trains at all.**

> This is the same class of problem as Day 1's softmax `- max` trick and Day 5's
> int32 accumulator: *keep the numbers inside the range your number type can hold.*
> Three different days, one recurring lesson.

### Why IDX parsing is worth doing by hand

```text
magic number → dimensions → raw pixel bytes, big-endian
```

You wrote a parser for a binary format with a magic number and a header. **That is
the same shape of work as Day 7's `weights.bin`** — and it's why designing your own
model format later felt natural rather than mysterious.

---

## 2. Train / test split — and what it actually caught

If you test a student on the exact questions they studied, you learn nothing about
whether they understood the subject.

```text
[70,000 MNIST images]
       │
       ├──► TRAIN ──► forward + backward. The network LEARNS from these.
       │
       └──► TEST  ──► forward ONLY. Never seen during training.
```

### ⚠️ This script uses a subset — and you should know why

```python
TRAIN_SIZE = 10000
TEST_SIZE  = 2000
```

Not the full 60k/10k. The script says why on the line below:

> *"Full 60k would take hours in pure NumPy — that's what PyTorch solves!"*

That's an honest and correct engineering decision. **But the summary docs originally
claimed 60,000 and 10,000**, which overstated the work. Corrected — see §6.

Being precise about this matters more than the bigger number would: an examiner who
spots an inflated claim stops trusting the rest of the file.

---

## 3. Results — read all four numbers

```text
BEFORE training:  train 8.1%   test 8.8%   [random guessing = 10.0%]

Final train accuracy: 100.0%
Final test accuracy:   95.0%

GENERALIZATION GAP: 5.0%
```

**8.1% before training.** Roughly chance for 10 classes, as it should be.

**100% on train, 95% on test.** The network memorized the training set *perfectly*
and is 5 points worse on data it never saw.

> **That 5-point gap IS overfitting**, and it is the entire reason a test set exists.
> Train accuracy alone would have told you the network was flawless. It isn't.
>
> The fix is regularization — dropout, weight decay, data augmentation, early
> stopping. None of that is in this file; it's Day 15 in the roadmap. Day 3's job is
> to make the gap *visible*.

### The "87%" trap — same bug as Day 2

The script used to print `Starting accuracy: 87.0%`. That was `train_accs[0]` — the
average over epoch 0, measured **after 10,000 weight updates**. Not a baseline.

The true starting point is **8.1%**. Fixed; see §6.

> **The general lesson: an "initial" metric recorded inside the first epoch is not
> initial.** Measure before the loop, not on its first iteration.

---

## 4. Per-digit accuracy — where it actually fails

```text
  Digit 0:  97.7%      Digit 5:  92.7%
  Digit 1:  99.1%      Digit 6:  94.9%
  Digit 2:  93.2%      Digit 7:  91.2%   ← worst
  Digit 3:  97.1%      Digit 8:  91.7%
  Digit 4:  96.8%      Digit 9:  94.3%
```

**Aggregate accuracy hides this.** 95% sounds uniform; it isn't. The network is
near-perfect on `1` (99.1%) and notably weaker on `7` (91.2%) and `8` (91.7%).

The pattern is not random — `7` is confusable with `1` and `9`, `8` with `3` and `5`.
Those are the digits humans confuse in bad handwriting too.

> **Why per-class analysis matters more than accuracy.** A model that is 99% accurate
> overall but 0% on one class is worthless for that class. On a detection system,
> "95% accurate" might mean "misses every person wearing dark clothing". Aggregate
> numbers hide systematic failure — always break them down.

---

## 5. Epochs, shuffling, and learning-rate decay

* **Epoch** — one complete pass through the training set.
* **Shuffling** — reorder before every epoch. Without it, if the network saw 1,000
  zeros then 1,000 ones, it would drift toward whatever it saw most recently
  (*catastrophic forgetting*). Shuffling forces it to hold all ten concepts at once.
* **Learning-rate decay** — this script multiplies `lr` by 0.7 every 5 epochs.

```text
Early training: big steps    — get near the valley fast
Late training:  small steps  — settle into the bottom instead of bouncing
```

Day 2's sweep showed a too-large `lr` overshoots the minimum. Decay gets both:
speed early, precision late. It's the cheapest accuracy improvement available.

---

## 6. 🐛 Two issues found and fixed

**1. The baseline accuracy was misreported.** `train_accs[0]` was printed as
"Starting accuracy" alongside "random = 10%", but it's the epoch-0 average after
10,000 updates. It read 87.0%; the true pre-training figure is **8.1%**. The script
now measures before the loop and prints both, plus the generalization gap.

*(This is the same bug that was in Day 2's `part3` — worth remembering as a pattern,
not a one-off.)*

**2. The dataset size was overstated in the docs.** `daily_log.md` and `README.md`
claimed 60,000 training and 10,000 test images. The script actually uses
**10,000 / 2,000**, deliberately and for a good reason. Corrected in both.

---

## 7. Where to use it & why

* **Phase:** data engineering — before any training happens.
* **Hardware:** CPU and RAM. Data loading is CPU work; the math is what goes to GPU.
* **The real lesson:** *garbage in, garbage out.* The most advanced architecture in
  the world fails on un-normalized, unshuffled data. This is the unglamorous half of
  machine learning and the half that decides whether anything works.

> In production this stage becomes a `DataLoader` with workers, prefetching and
> augmentation. Same pipeline, more engineering. Having built it by hand once means
> you'll know what `torchvision.transforms.Normalize` is actually doing.

---

## ✅ Self-test

1. Why does skipping `/255` break training? What exactly overflows?
2. Train 100%, test 95%. What is that gap called, and why is it not a bug in the code?
3. Your untrained 10-class network scores 8.1%. Is that a problem?
4. Why is an accuracy figure recorded during epoch 0 not a baseline?
5. Overall accuracy is 95%. Why is per-digit accuracy worth computing anyway?
6. Why shuffle before every epoch rather than once at the start?
7. Why decay the learning rate instead of picking one good value?

<details><summary>Answers</summary>

1. Inputs up to 255 across 784 terms make logits huge; `exp()` in softmax overflows to `inf`, and gradients become `inf`/`NaN` on the first step.
2. Overfitting. The network memorized the training set. It's not a code bug — it's what a test set exists to reveal, and it's fixed with regularization, not debugging.
3. No — that's about chance level for 10 classes (10%), exactly what an untrained network should score.
4. By the end of epoch 0 the network has already taken one gradient step per sample — 10,000 of them. It's an average *while learning*, not a starting point.
5. Aggregates hide systematic failure. Here `7` is at 91.2% and `1` at 99.1% — an 8-point spread invisible in the headline number.
6. Once isn't enough: every epoch would replay the same order, so the network could learn order-dependent patterns. Reshuffling makes each pass an independent sample of the ordering.
7. Big steps converge fast but bounce around the minimum; small steps settle precisely but crawl. Decay gets speed early and precision late.

</details>

---

**Next:** `day4/` — the same network in C, where you manage memory yourself and
discover that RAM is a single flat line of bytes.
