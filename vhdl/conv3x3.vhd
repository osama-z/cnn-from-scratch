-- =====================================================================
-- conv3x3.vhd — 3x3 convolution, nine multiplies in parallel
-- =====================================================================
--
-- WHAT THIS REPLACES
-- ------------------
-- In C (day7/golden_c.c) the inner convolution is four nested loops:
--
--     for ky in -1..1:
--       for kx in -1..1:
--         sum += in[...] * kern[...];
--
-- Nine iterations, executed one after another, each costing at least a
-- multiply-accumulate instruction. On a Cortex-M4 that is ~9 cycles minimum
-- for one output pixel.
--
-- Here all nine multiplies happen AT THE SAME TIME, in parallel silicon, and
-- an adder tree sums them in the same clock cycle. One output pixel per cycle.
--
-- THIS IS THE WHOLE ARGUMENT FOR FPGA ACCELERATION:
-- the CPU has one multiplier and must reuse it nine times; the FPGA has nine
-- multipliers and uses them once. Same arithmetic, one ninth the latency.
--
-- WHY FULLY PARALLEL AND NOT A LINE BUFFER
-- ----------------------------------------
-- A production engine adds a line buffer so pixels stream in one at a time and
-- windows are formed on the fly, saving input bandwidth. That is a real
-- optimisation and a separate problem.
--
-- This version takes the window as a port and leaves window formation to the
-- caller. It is deliberately the simpler design, because it still verifies
-- bit-exactly against the golden model -- and a verified simple engine is worth
-- more than an unverified clever one. The line buffer can be added later
-- without touching this arithmetic.
--
-- COMBINATIONAL, NOT CLOCKED
-- --------------------------
-- There is no clock here. Given a window and a kernel, the output settles after
-- propagation delay through the multipliers and the adder tree. That makes it
-- easy to verify (no timing to reason about) and easy to pipeline later: adding
-- registers between tree stages is a local change, and it is what raises Fmax
-- when the critical path through this adder tree becomes the limit.

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.cnn_types.all;

entity conv3x3 is
    generic (
        ACC_WIDTH : positive := 32       -- see mac_unit.vhd: 48,387 needs 17 bits
    );
    port (
        window : in  int8_array(0 to 8);              -- 3x3 patch, row-major
        kernel : in  int8_array(0 to 8);              -- 3x3 weights, row-major
        acc    : out signed(ACC_WIDTH-1 downto 0)     -- int32 sum of nine products
    );
end entity conv3x3;

architecture rtl of conv3x3 is
begin

    process (window, kernel)
        -- Each product is 16 bits (8x8). The running sum is ACC_WIDTH so it
        -- cannot overflow -- the same discipline as mac_unit, and the reason
        -- resize() appears below rather than a bare addition.
        variable sum : signed(ACC_WIDTH-1 downto 0);
    begin
        sum := (others => '0');

        -- A `for` loop in a combinational process is UNROLLED by the
        -- synthesiser: it does not become a sequential loop in hardware, it
        -- becomes nine multipliers and an adder tree. This is the single
        -- biggest mental shift from software -- the loop is a description of
        -- structure, not a description of time.
        for i in 0 to 8 loop
            sum := sum + resize(window(i) * kernel(i), ACC_WIDTH);
        end loop;

        acc <= sum;
    end process;

end architecture rtl;
