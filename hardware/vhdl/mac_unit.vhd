-- =====================================================================
-- mac_unit.vhd — Multiply-Accumulate, int8 x int8 -> int32
-- =====================================================================
--
-- THIS IS DAY 5'S RULE AS PHYSICAL HARDWARE.
--
-- Day 5 (C)   : "remember to accumulate in int32"   -- a comment
-- Day 6 (C++) : AccumTraits<int8_t> = int32_t       -- the compiler enforces it
-- Here (VHDL) : a 32-bit register                   -- a physical wire
--
-- WHY 32 BITS AND NOT 16?
-- -----------------------
-- One product of two int8 values fits comfortably in 16 bits:
--     127 * 127  = 16,129
--    -128 * -128 = 16,384      (16-bit signed holds up to 32,767)
--
-- But convolution ACCUMULATES nine of them, and the golden model measured
-- the real peak on real data:
--
--     peak |accumulator| = 48,387   -> needs 17 bits + sign
--
-- 48,387 does not fit in 16 bits. So a 16-bit accumulator would silently
-- wrap and produce garbage, exactly as an int8 accumulator does in C.
-- 32 bits is not conservatism here; it is the measured requirement, taken
-- from lessons/day7/vhdl_vectors/quant_params.txt.
--
-- WHY `signed` AND NOT `std_logic_vector`?
-- ----------------------------------------
-- std_logic_vector is just a bag of bits with no numeric meaning -- the "*"
-- operator is not even defined for it. `signed` (from numeric_std) carries
-- two's-complement semantics, so "*" sign-extends correctly and the
-- synthesiser infers a signed multiplier or a DSP block. Using
-- std_logic_vector here and casting everywhere is how sign-extension bugs
-- get in.
--
-- INTERFACE NOTE
-- --------------
-- The accumulator is cleared by `clr`, not by `rst`. That separation
-- matters: rst is a global power-on reset, while clr fires once per output
-- pixel, mid-stream, thousands of times a second. Conflating them is a
-- classic source of "the first pixel of every row is wrong" bugs.

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity mac_unit is
    generic (
        DATA_WIDTH : positive := 8;    -- int8 operands
        ACC_WIDTH  : positive := 32    -- int32 accumulator (see header)
    );
    port (
        clk     : in  std_logic;
        rst     : in  std_logic;                                -- async power-on reset
        clr     : in  std_logic;                                -- clear accumulator (per output pixel)
        en      : in  std_logic;                                -- accumulate this cycle
        a       : in  signed(DATA_WIDTH-1 downto 0);            -- activation
        b       : in  signed(DATA_WIDTH-1 downto 0);            -- weight
        acc     : out signed(ACC_WIDTH-1 downto 0)              -- running sum
    );
end entity mac_unit;

architecture rtl of mac_unit is

    -- The accumulator register. This IS the int32 from Day 5, in silicon.
    signal acc_reg : signed(ACC_WIDTH-1 downto 0) := (others => '0');

begin

    -- Continuous assignment: the register drives the output port directly.
    acc <= acc_reg;

    process (clk, rst)
        -- The raw product is exactly 2*DATA_WIDTH bits wide. VHDL's "*" on
        -- signed returns that width, and naming it in a variable makes the
        -- widening explicit rather than hidden inside an expression.
        variable product : signed(2*DATA_WIDTH-1 downto 0);
    begin
        if rst = '1' then
            acc_reg <= (others => '0');

        elsif rising_edge(clk) then
            if clr = '1' then
                -- Clear takes priority over accumulate, so a clr and en in the
                -- same cycle starts a fresh sum rather than adding to stale data.
                acc_reg <= (others => '0');

            elsif en = '1' then
                product := a * b;

                -- resize() sign-extends the 16-bit product to 32 bits.
                -- Without it the addition would be width-mismatched and GHDL
                -- would reject it -- VHDL is strict about widths on purpose,
                -- because in hardware a width mismatch is a real wiring error,
                -- not something to paper over.
                acc_reg <= acc_reg + resize(product, ACC_WIDTH);
            end if;
        end if;
    end process;

end architecture rtl;
