# 🚀 CNN From Scratch — Python → C → ARM → FPGA → Drone

> Building a real-time human detection system from absolute scratch.
> No frameworks until Month 2. Pure math, pure understanding.

**Author:** Osama Z.
**Profile:** Hardware Engineer mastering AI — VHDL + C + Python + Neural Networks

---

## 📊 Progress

```text
Month 1: Build It By Hand
████████████░░░░░░░░░░░░░░░░░░  Day 5 / 21

Week 1 (Python)  ████████████████████  DONE — NumPy CNN + 95% MNIST
Week 2 (C)       ████████████░░░░░░░░  Day 5/7 — int8 quantization done
Week 3 (ARM)     ░░░░░░░░░░░░░░░░░░░░  NOT STARTED
Week 4 (Review)  ░░░░░░░░░░░░░░░░░░░░  NOT STARTED
```

---

## 🏆 Key Results

| Day | Achievement | Numbers |
|-----|------------|---------|
| Day 1 | CNN forward pass from scratch | Conv→ReLU→Pool→Dense→Softmax |
| Day 2 | Training loop with 3 optimizers | 97%→100% on toy data |
| Day 3 | MNIST digit classification | **95% test accuracy** |
| Day 4 | Full CNN rewritten in C | **16,000+ FPS**, 1.4 KB memory |
| Day 5 | int8 quantization | **4x smaller**, 20x faster |

---

## 📁 Structure

```text
projects/
├── README.md              ← This file
├── ROADMAP.md             ← Full 6-month plan (v4.0 — CE tailored)
├── daily_log.md           ← Learning journal
├── notes_week1.md         ← Formula cheat sheet
├── day1/                  ✅ CNN Forward Pass (Python)
├── day2/                  ✅ Loss + Backprop + Training (Python)
├── day3/                  ✅ MNIST 95% Accuracy (Python)
├── day4/                  ✅ CNN Forward Pass (C)
└── day5/                  ✅ Fixed-Point int8 (C)
```

Each day folder contains:
- **Code files** — Heavily commented Python/C
- **README_explanation.md** — Deep visual explanation with ASCII diagrams

---

## 🛠️ How to Run

```bash
# Python (Days 1-3)
cd ai-workspace && source venv/bin/activate
cd projects/day3 && python part1_mnist_classifier.py

# C (Days 4-5)
cd projects/day4 && make && ./cnn_forward
cd projects/day5 && make && ./fixed_point
```
