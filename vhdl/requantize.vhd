-- =====================================================================
-- requantize.vhd — int32 accumulator back down to int8
-- =====================================================================
--
-- THE PROBLEM
-- -----------
-- The MAC unit produces an int32 accumulator carrying units of
-- (input_scale * weight_scale). To store the result as int8 it must be
-- rescaled into the OUTPUT tensor's units:
--
--     M = (input_scale * weight_scale) / output_scale
--     out = saturate( round(acc * M) + output_zero_point )
--
-- In C (day6/templated_cnn.cpp) M is a float and that is the end of it.
--
-- WHY HARDWARE CANNOT DO THAT
-- ---------------------------
-- There is no float multiplier in this design, and putting one there would
-- defeat the entire purpose -- the reason for int8 is to avoid floating-point
-- hardware. On a small FPGA a float multiplier costs hundreds of LUTs; an
-- integer multiply plus a shift costs a DSP block or a few dozen LUTs.
--
-- THE SOLUTION: FIXED-POINT MULTIPLIER
-- ------------------------------------
-- Approximate M as an integer over a power of two:
--
--     M  ~=  M0 / 2^SHIFT          M0 and SHIFT are integers
--
-- Then "multiply by M" becomes "multiply by M0, then shift right by SHIFT".
-- Both are cheap. This is exactly what TFLite Micro and CMSIS-NN do -- look
-- for `quantized_multiplier` and `shift` in their source.
--
-- For this network:
--     M = (10/127 * 1/127) / (30/127) = 0.002624671916
--     M0 = 172, SHIFT = 16  ->  172/65536 = 0.002624511719   (0.006% error)
--
-- And the number that matters:
--     acc = 48,387 (the measured peak)  ->  48387 * 172 >> 16 = 127
-- The worst-case accumulator maps exactly onto int8's maximum. That is not a
-- coincidence -- output_scale was chosen so the range fits.
--
-- WHY SATURATION AND NOT WRAPPING
-- -------------------------------
-- If a result exceeds +127 it must CLAMP to +127, never wrap to -128. Wrapping
-- turns "very bright" into "very dark" -- a sign inversion in the middle of a
-- feature map. Saturating arithmetic is standard in DSP hardware for exactly
-- this reason, and it is why C's saturate_i8() exists in day6.
--
-- ROUNDING
-- --------
-- A plain right shift truncates toward negative infinity, which biases every
-- output downward. Adding half the divisor first gives round-half-away-from-zero,
-- matching the C reference's std::lround(). The sign is handled explicitly
-- because shifting a negative number rounds the wrong way.

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity requantize is
    generic (
        ACC_WIDTH  : positive := 32;   -- accumulator input width
        DATA_WIDTH : positive := 8;    -- int8 output
        MULT_WIDTH : positive := 32    -- width of the fixed-point multiplier M0
    );
    port (
        acc        : in  signed(ACC_WIDTH-1 downto 0);    -- from the MAC
        multiplier : in  signed(MULT_WIDTH-1 downto 0);   -- M0
        shift      : in  natural range 0 to 63;           -- SHIFT
        zero_point : in  signed(DATA_WIDTH-1 downto 0);   -- output zero point
        y          : out signed(DATA_WIDTH-1 downto 0)    -- saturated int8
    );
end entity requantize;

architecture rtl of requantize is

    -- The product needs the FULL combined width. acc is 32 bits and M0 is 32
    -- bits, so the product is up to 64 bits. Declaring it any narrower would
    -- silently truncate -- the same overflow mistake as an int8 accumulator,
    -- one level further along.
    constant PROD_WIDTH : positive := ACC_WIDTH + MULT_WIDTH;

    constant MAX_I8 : integer := 2**(DATA_WIDTH-1) - 1;   --  127
    constant MIN_I8 : integer := -(2**(DATA_WIDTH-1));    -- -128

begin

    process (acc, multiplier, shift, zero_point)
        variable product : signed(PROD_WIDTH-1 downto 0);
        variable half    : signed(PROD_WIDTH-1 downto 0);
        variable shifted : signed(PROD_WIDTH-1 downto 0);
        variable total   : integer;
    begin
        product := acc * multiplier;

        -- Half of 2^shift, for round-half-away-from-zero.
        -- shift = 0 is a special case: there is nothing to round.
        if shift = 0 then
            half := (others => '0');
        else
            half := to_signed(1, PROD_WIDTH) sll (shift - 1);
        end if;

        -- Shift toward zero on both sides of zero, so positive and negative
        -- values round symmetrically. A bare `shift_right` on a negative number
        -- rounds toward negative infinity and would bias the output.
        if product >= 0 then
            shifted := shift_right(product + half, shift);
        else
            shifted := -shift_right((-product) + half, shift);
        end if;

        total := to_integer(shifted) + to_integer(zero_point);

        -- Saturate. Never wrap: wrapping would flip bright to dark.
        if total > MAX_I8 then
            y <= to_signed(MAX_I8, DATA_WIDTH);
        elsif total < MIN_I8 then
            y <= to_signed(MIN_I8, DATA_WIDTH);
        else
            y <= to_signed(total, DATA_WIDTH);
        end if;
    end process;

end architecture rtl;
