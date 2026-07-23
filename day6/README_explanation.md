# 🧩 Day 6: C++ for AI Deployment (RAII, Polymorphism, Templates)

> **Core Concept:** Day 4 proved C is fast. But every production inference engine
> — TensorFlow Lite, ONNX Runtime, LibTorch, TensorRT, ncnn — is written in **C++**,
> not C. This day is about learning why, and rebuilding the CNN the way those
> engines are actually structured.

---

## 1. Why C++ and Not Just C?

You already have a working CNN in C. So why rewrite it?

```text
Every production AI inference engine, and what it's written in:

  TensorFlow Lite   ->  C++
  ONNX Runtime      ->  C++
  PyTorch LibTorch  ->  C++
  TensorRT (NVIDIA) ->  C++
  ncnn (Tencent)    ->  C++
  CMSIS-NN          ->  C  (the exception: bare-metal MCU kernels)
```

To read their source, debug them, or extend them, you need C++. Not the whole
language — three ideas carry most of the weight: **RAII**, **polymorphism**, and
**templates**. This day builds one CNN with each.

**Files:**
- `cnn_forward.cpp` — part 1: classes, inheritance, smart pointers
- `templated_cnn.cpp` — part 2: one source, two number types

---

## 2. `malloc`/`free` → RAII

In C, every allocation is a promise you must personally keep:

```c
/* Day 4 */
float* buf = (float*)malloc(n * sizeof(float));
if (!buf) return -1;

if (something_failed) {
    return -1;              /* ← LEAK. You just skipped free(). */
}

free(buf);
```

That early `return` is the classic embedded bug: not a crash, just a slow drip
of RAM until the drone falls out of the sky an hour into the flight.

In C++ the resource belongs to an **object**, and the object's destructor runs
automatically at the closing brace:

```cpp
/* Day 6 */
std::vector<float> buf(n);   // constructor allocates

if (something_failed) {
    return -1;               // ← destructor runs. No leak. Cannot be forgotten.
}
```

**RAII = Resource Acquisition Is Initialization.** Acquire in the constructor,
release in the destructor, tie both to a scope.

The critical part is *every exit path*: normal return, early return, `break`,
and — the one C cannot handle at all — a thrown exception. The compiler emits
the destructor call on all of them.

### Verified, not asserted

`cnn_forward.cpp` prints `Memory leaks: Impossible (RAII)`. That is a claim, so
it should be tested:

```bash
make asan
```

```text
--- running cnn_forward_asan ---
    clean: no leaks, no UB
--- running templated_cnn_asan ---
    clean: no leaks, no UB
```

Both programs allocate constantly — six layers via `make_unique`, plus a fresh
`Tensor` returned from every single `forward()` call — and contain **zero
`free()` and zero `delete`**. LeakSanitizer finds nothing at exit.

> *(Uses `-fsanitize=address,undefined`, built into GCC. Valgrind is not
> required and is not installed on this machine.)*

---

## 3. The Tensor Class — Hiding the Index Arithmetic

Day 4's indexing, everywhere a tensor was touched:

```c
data[c * H * W + y * W + x]
```

The formula is correct, but it is written out at every use site. Get one wrong
and you read a neighbouring channel — no crash, just subtly wrong numbers.

```cpp
float& operator()(int c, int y, int x) {
    return data[c * (height * width) + y * width + x];
}
```

Now it is written **once**:

```text
    C   (Day 4):   output[f*H*W + y*W + x] = sum;
    C++ (Day 6):   output(f, y, x) = sum;
```

Identical machine code after `-O2`. The win is not speed — it is that a bug in
the index math has exactly one place to hide.

---

## 4. Inheritance + Virtual Functions — One Interface, Six Layers

Day 4 called each layer by name, in order, with hand-managed buffers:

```c
conv2d(input, ..., conv_out);
relu(conv_out, size);
maxpool2d(conv_out, ..., pool_out);
dense(pool_out, ..., dense_out);
softmax_inplace(dense_out, n);
```

Adding a layer means editing this sequence and allocating another buffer.

C++ gives every layer the same interface:

```cpp
class Layer {
public:
    virtual Tensor forward(const Tensor& input) = 0;   // "= 0" -> pure virtual
    virtual ~Layer() = default;                        // required for cleanup
};
```

`= 0` makes it **pure virtual**: any class inheriting `Layer` *must* implement
`forward()` or the program will not compile. Forgetting becomes a build error
instead of a runtime surprise.

Now the entire network is one loop:

```cpp
for (auto& layer : layers)
    x = layer->forward(x);
```

```text
        ┌─────────────────────────────┐
        │   Layer  (abstract)         │
        │   virtual forward() = 0     │
        └──────────────┬──────────────┘
                       │
   ┌────────┬──────────┼──────────┬─────────┬─────────┐
   ▼        ▼          ▼          ▼         ▼         ▼
 Conv2D   ReLU    MaxPool2D   Flatten    Dense    Softmax

   Six different implementations, one calling convention.
```

**Do not forget the virtual destructor.** Deleting a derived object through a
`Layer*` when the destructor is non-virtual is undefined behaviour — in
practice, the derived part never gets destroyed, and you leak.

---

## 5. `std::unique_ptr` — Ownership You Cannot Duplicate

```cpp
std::vector<std::unique_ptr<Layer>> layers;

void add(std::unique_ptr<Layer> layer) {
    layers.push_back(std::move(layer));
}
```

A `unique_ptr` owns its object and frees it when destroyed. It cannot be copied
— only **moved** — which is why `std::move` is required. That restriction is the
feature: it makes double-free impossible to express.

```text
  cnn.add(std::make_unique<Conv2D>(1, 2, 3));
       │
       └─► ownership transfers into the vector; the caller's pointer is now empty

  When CNN is destroyed:
       vector destructor
         └─► each unique_ptr destructor
               └─► virtual ~Layer()
                     └─► ~Conv2D(), freeing its weight vectors

  Every step automatic. No delete anywhere in the file.
```

---

## 6. Templates — One Source, Two Number Types

Day 5 built an int8 CNN in C. Day 6 part 1 built a float CNN in C++. Naively you
now need **two copies of every layer** — and two copies drift apart the moment
you fix a bug in one.

Templates let you write each layer once:

```cpp
template <typename T>
class Conv2D : public Layer<T> { ... };

CNN<float>  fnet;   // float32 inference
CNN<int8_t> qnet;   // int8 inference
```

The compiler stamps out a separate, fully specialized class for each type you
actually use. `Tensor<float>` and `Tensor<int8_t>` are two unrelated types by
the time the program runs.

### Why the whole class is templated, not just the method

The obvious attempt is illegal:

```cpp
class Layer {
    template <typename T>
    virtual Tensor<T> forward(const Tensor<T>&) = 0;   // ← will not compile
};
```

**C++ forbids a member function from being both `virtual` and a template.**

The reason is mechanical. A polymorphic class has a **vtable** — a fixed-size
array of function pointers, laid out at compile time. Templates are instantiated
on demand, so a virtual template would need one slot per type anyone ever
instantiates it with, including types in files not yet written. The table has no
finite size, so the compiler cannot build it.

So `Layer<float>` and `Layer<int8_t>` are two separate classes, each with its own
ordinary vtable. This is a real language constraint, and knowing *why* it exists
is the difference between "I used C++" and "I understand C++".

---

## 7. 🔑 The Trap Templates Solve — The Day 5 → Day 6 Bridge

**This is the most important section of the day.**

Day 5 (`day5/README_explanation.md`, section 4) established the rule:

```text
int8 range: -128 .. 127

    127 * 127 = 16,129            ← already too big for int8
    16,129 * 9 = 145,161          ← a 3x3 kernel sums nine of them

  => int8 x int8 MUST accumulate in int32.
```

On Day 5 that was a rule you had to **remember** while writing C. Forget it and
there is no crash and no warning — just quietly wrong predictions.

Now here is the trap. Templating the storage type does **not** fix this:

```cpp
template <typename T>
Tensor<T> forward(const Tensor<T>& input) {
    T acc = 0;                    // ← T is int8_t. OVERFLOW. Exactly Day 5's bug,
    acc += input[i] * w[i];       //   now generated automatically for you.
}
```

The accumulator must be a **different type** from the storage. That is what a
trait does:

```cpp
template <typename T> struct AccumTraits           { using type = float;   };
template <>           struct AccumTraits<int8_t>   { using type = int32_t; };

template <typename T> using Accum = typename AccumTraits<T>::type;
```

A **trait** is a compile-time lookup table: give it a type, get back another
type. Now the accumulator is chosen by the compiler:

```cpp
Accum<T> acc = biases[oc];    // float  -> float
                              // int8_t -> int32_t
```

```text
   Day 5 (C)      "remember to use int32"      a comment you must obey
   Day 6 (C++)    Accum<int8_t> = int32_t      the compiler obeys it for you
   Phase 1 (VHDL) a 32-bit accumulator reg     the rule as a physical datapath
```

Same rule, three levels of abstraction. That progression connects numerical
requirements, software types, and hardware design.

The file also states the rule as a compile-time assertion, so a future
"simplification" breaks the build instead of the predictions:

```cpp
static_assert(std::is_same_v<Accum<int8_t>, int32_t>,
              "int8 network MUST accumulate in int32 -- see day5 section 4");
```

This storage-type/accumulator-type split is not a teaching invention — CMSIS-NN
and TFLite Micro carry it through all their kernels.

### `if constexpr` — the branch that costs nothing

Two places genuinely differ between float and int8. **Requantization:**

```cpp
if constexpr (is_quantized_v<T>)
    output(oc,y,x) = requantize(acc, M, out_q.zero_point);
else
    output(oc,y,x) = acc;
```

And **ReLU** — Day 5's zero-point trap (`day5/README_explanation.md` §5):

```cpp
if constexpr (is_quantized_v<T>) {
    const T zp = static_cast<T>(out.q.zero_point);
    if (out[i] < zp) out[i] = zp;      // max(zero_point, x)
} else {
    if (out[i] < 0) out[i] = 0;        // max(0.0, x)
}
```

`if constexpr` is resolved **while compiling**. The discarded branch is not
compiled into a dead branch or a predicted-away jump — it is never emitted.

Proof, from `make asm`:

| Instruction | `Conv2D<float>` | `Conv2D<int8_t>` |
|---|---|---|
| `mulss` — scalar FP multiply | ✅ present | ❌ absent |
| `movsbl` — sign-extend **byte → long** | ❌ absent | ✅ present |
| `imull` + `addl` — 32-bit MAC | ❌ absent | ✅ present |

The two instantiations share **zero** arithmetic instructions. `movsbl` is
literally the int8 → int32 widening that `AccumTraits` demanded, appearing in the
machine code.

> **"One code path" is a convenience for the reader. It is never a cost for the CPU.**

### Does the int8 network actually agree with the float one?

```text
    image             float32 probabilities     int8 probabilities      max err
    -------------------------------------------------------------------------
    Horizontal Edge   [100.0,   0.0,   0.0]     [100.0,   0.0,   0.0]     0.00%
    Vertical Edge     [  0.2,   0.0,  99.7]     [  0.4,   0.0,  99.6]     0.15%
    Uniform           [  0.5,  56.5,  43.1]     [  0.4,  55.3,  44.3]     1.26%

    Predictions agreeing: 3 / 3
    Weight memory: 456 bytes float32 -> 114 bytes int8   (4x smaller)
```

Largest disagreement: **1.26 percentage points** — consistent with the ~1.5%
accuracy cost Day 5 predicted. Quantization error is real but small, and it did
not change a single decision.

> ⚠️ **The class labels are meaningless, and that is expected.** The conv kernels
> are hand-designed edge detectors, but the Dense layer is generated and never
> trained, so it maps good features to arbitrary classes. Day 6 tests that *two
> number systems agree*, not that the network is accurate. Training was Day 2's
> job. Read the columns against each other, not against the labels.

---

## 8. C (Day 4) vs C++ (Day 6)

| Aspect | C (Day 4) | C++ (Day 6) |
| :--- | :--- | :--- |
| Memory management | `malloc` / `free` | RAII — automatic |
| Leak on early return | Very possible | Impossible |
| Data structure | `float*` + separate dims | `Tensor<T>` |
| Index arithmetic | Repeated at each use | Once, in `operator()` |
| Layer interface | Distinct function per layer | `virtual forward()` |
| Adding a layer | Edit call sequence + buffers | `cnn.add(...)` |
| float + int8 | Two separate files | One templated source |
| Accumulator safety | A rule you remember | `static_assert` + traits |
| Runtime speed | Baseline | Identical after `-O2` |
| Binary size | ~25 KB | ~50 KB |
| Used by TFLite / ONNX RT | No | **Yes** |

**C++ is not slower than C.** Templates and `if constexpr` are resolved at
compile time; `operator()` and non-virtual calls inline away. You pay in binary
size and compile time, not cycles.

The one real runtime cost is **virtual dispatch** — an indirect call through the
vtable, which cannot be inlined. Per *layer*, not per element, so it is
irrelevant here. On a Cortex-M with millions of calls it would matter, which is
exactly why CMSIS-NN is plain C.

---

## 9. Where to Use It & Why

* **The Phase:** Deployment — writing or extending the inference engine itself.
* **The Hardware:** Anything running an OS. Raspberry Pi, Jetson, phones, servers.
* **When to drop back to C:** Bare-metal Cortex-M. No exceptions, no heap, no STL.
  That is CMSIS-NN's world, and why it is C.
* **The Workflow:**
  1. Train in Python (Month 2)
  2. Export to ONNX / TFLite (Month 3)
  3. The **C++** runtime loads it and executes the graph
  4. Templates let that one runtime serve float32 *and* int8 targets

> **The Golden Rule:** C for the kernel, C++ for the engine.
> CMSIS-NN's `arm_convolve_s8` is C. TFLite Micro, which *calls* it, is C++.

---

## 10. ⚠️ Known Issue Waiting in Day 7

Day 7 requires verifying that Python, C, and C++ produce identical output
("golden model" testing). **Right now they cannot.**

| | Day 4 C (`day4/cnn_forward.c:384`) | Day 6 C++ (`day6/cnn_forward.cpp:321`) |
|---|---|---|
| Dense weights | `((rand()%100)/100.0f - 0.5f) * 0.1f` | `(rand()/RAND_MAX - 0.5f) * 2.0f * scale` |
| Dense bias | `{0.1f, -0.1f, 0.0f}` | all `0.0f` |

Same architecture, different numbers. Both call `srand(42)`, but **seeding two
different formulas does not produce the same sequence** — the seed only makes
each program reproducible against *itself*.

**The Day 7 fix:** one shared weights file that Python, C, and C++ all *load*
instead of generating. That is worth doing properly, because you will be
inventing a **model file format** by hand — which is what ONNX and TFLite are.
Doing it yourself once means those tools look obvious later instead of magic.

`templated_cnn.cpp` already takes the first step: it uses a deterministic,
closed-form weight function instead of `rand()`, so its numbers are reproducible
across compilers and machines.

---

## ✅ Self-Test

1. What is RAII, and why does it prevent memory leaks that C cannot?
2. Why can't `forward()` be a virtual template function?
3. Why doesn't `Tensor<int8_t>` alone fix the overflow problem?
4. In int8, why is `relu(x) = max(0, x)` wrong?
5. What does `if constexpr` cost at runtime? *(Answer: nothing — check `make asm`.)*
