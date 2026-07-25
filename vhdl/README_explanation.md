# ⚡ Phase 1: The Arithmetic Becomes Hardware (VHDL)

> **Core Concept:** Day 5 taught "accumulate in int32" as a rule to remember.
> Day 6 made the compiler enforce it. Here it becomes a **32-bit register** — a
> physical thing, whose width is decided by a number measured from real data.

---

## 1. Where this sits in the project

```text
        int8 × int8 overflows → you MUST accumulate in int32

Day 5    a comment you have to remember             (C)
Day 6    a type the compiler enforces               (C++ AccumTraits)
Day 7    48,387 — the peak, measured on real data   (golden model)
HERE     a 32-bit accumulator register              (hardware)
```

Every earlier layer of the project exists to make this file correct. The
accumulator width isn't a guess — `day7/vhdl_vectors/quant_params.txt` records it:

```text
peak_abs_accumulator 48387
accumulator_bits_required 17
```

---

## 2. Why 32 bits and not 16

One product of two int8 values fits easily in 16 bits:

```text
 127 ×  127 = 16,129
-128 × -128 = 16,384        16-bit signed holds up to 32,767  ✓
```

But convolution **accumulates nine of them**:

```text
9 × 16,384 = 147,456        needs 18 bits
real measured peak = 48,387 needs 17 bits + sign
```

So a 16-bit accumulator silently wraps — exactly the way an int8 accumulator does
in C. This was tested rather than assumed; see §5.

---

## 3. Two VHDL details that matter

### `signed`, not `std_logic_vector`

`std_logic_vector` is a bag of bits with no numeric meaning — `*` isn't even
defined for it. `signed` (from `numeric_std`) carries two's-complement semantics,
so `*` sign-extends correctly and the synthesiser infers a signed multiplier or a
DSP block.

Using `std_logic_vector` and casting at every use site is how sign-extension bugs
get in.

### `clr` is separate from `rst`

```vhdl
rst : in std_logic;   -- global power-on reset
clr : in std_logic;   -- clear the accumulator, once per output pixel
```

`rst` fires once at power-on. `clr` fires **thousands of times a second** —
once per output pixel, mid-stream. Conflating them produces the classic
"first pixel of every row is wrong" bug.

`clr` also takes priority over `en`, so both asserted in the same cycle starts a
fresh sum instead of adding to stale data.

### Explicit width management

```vhdl
variable product : signed(2*DATA_WIDTH-1 downto 0);   -- 8×8 → 16 bits
...
acc_reg <= acc_reg + resize(product, ACC_WIDTH);      -- 16 → 32, sign-extended
```

VHDL refuses to add mismatched widths. That strictness is deliberate: in hardware
a width mismatch is a real wiring error, not something to paper over silently.

---

## 4. Two errors worth knowing about

Both hit on the first compile:

**`label` is a reserved word.** It names statement labels in VHDL, so it can't be
a parameter name. Renamed to `test_name`.

**String literals accept only the basic character set.** An em-dash (`—`) inside
`report "..."` is a hard error. Comments can contain anything; string literals
cannot.

---

## 5. Testing the testbench

Same discipline as Day 7: a test that cannot fail proves nothing. The accumulator
width was shrunk from 32 to 16 bits to check the testbench notices.

```text
PASS  127*127: 16129                                  ← still passes (fits)
PASS  -128*-128: 16384                                ← still passes (fits)
PASS  3 x (100*100): 30000                            ← still passes (fits)
FAIL  9 x (-128*-128): got 16384, expected 147456     ← WRAPPED
exit code 2
```

`147,456 mod 65,536 = 16,384` — precisely the predicted wraparound.

**The important part: the first three checks still passed.** Small operands hide
overflow. Only the worst case exposes it — which is the same lesson the Day 7
softmax hole taught, in different clothes:

> **A test built from convenient values tests convenient behaviour.**

This is also why `--assert-level=error` is in the Makefile: without it a failing
`report` prints and the simulation still exits 0, so a broken design would look
like a passing build.

---

## 6. What the testbench checks

| # | Test | Why |
|---|---|---|
| 1 | `127×127`, `-128×-128`, `-128×127` | The extremes — values that overflow int8 |
| 2 | Three accumulations in sequence | The accumulator actually accumulates |
| 3 | Nine worst-case products = 147,456 | Proves 16 bits is insufficient |
| 4 | `clr` between two sums | No state leaks into the next pixel |
| 5 | **One real 3×3 convolution window → 48,387** | **Matches the golden model** |

Test 5 is the point. Tests 1–4 confirm the arithmetic works; test 5 confirms this
hardware computes *the same thing* NumPy, C and C++ already agree on. The window
is image 0 at (y=4, x=4), filter 0 — and it happens to be the worst case in the
whole dataset, which is why it sizes the register.

---

## 7. Run it

```bash
cd vhdl
make test     # analyze, elaborate, simulate, check all 7 assertions
make wave     # same + dump a waveform:  gtkwave build/mac.ghw
make synth    # prove it is synthesisable, not just simulatable
make clean
```

`make wave` is worth doing once. Watching the accumulator climb across nine clock
cycles is the fastest way to see that a MAC is a *sequential* circuit — it has
state, unlike the C function it replaces.

### Why `make synth` matters

GHDL's synthesis pass rejects constructs that simulate fine but cannot become
hardware. It's a cheap gate before anything goes near Vivado, and it's the
difference between "my VHDL runs" and "my VHDL is a circuit".

---

## 8. What GHDL actually does

```text
ghdl -a   ANALYZE     parse and type-check into the work library
ghdl -e   ELABORATE   link an entity and its children into a design
ghdl -r   RUN         simulate
```

Three steps, mirroring a real toolchain. Note `-fsynopsys` is deliberately **not**
used — it enables the non-standard `std_logic_arith` library, and depending on it
teaches habits that break on real synthesis tools. `numeric_std` is the standard.

---

## 9. Next

- [x] `mac_unit.vhd` — int8 × int8 → int32, verified against the golden model
- [ ] `relu_int8.vhd` — `max(zero_point, x)`, not `max(0, x)` (Day 5 §5)
- [ ] `requantize.vhd` — int32 → int8 with saturation
- [ ] `conv3x3_parallel.vhd` — nine MACs, verified against `image*_conv_acc_int32.txt`

The vectors for that last step already exist. `export_weights.py` wrote them when
the golden model was built, which was the whole reason for building it first.

---

## ✅ Self-test

1. Why is a 16-bit accumulator wrong when a single int8 product fits in 16 bits?
2. Why `signed` instead of `std_logic_vector`?
3. Why are `rst` and `clr` separate ports?
4. Three of five checks passed with a 16-bit accumulator. What does that say about tests built from small values?
5. What does `resize()` do, and why won't VHDL let you omit it?
