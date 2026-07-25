-- =====================================================================
-- relu_int8.vhd — ReLU on quantized int8 data
-- =====================================================================
--
-- DAY 5'S TRAP, IN LOGIC GATES.
--
-- day5/README_explanation.md section 5:
--
--     float:  relu(x) = max(0.0, x)
--     int8 :  relu(x) = max(zero_point, x)        <-- NOT max(0, x)
--
-- The integer 0 does not represent the real value 0.0. The ZERO POINT does.
-- Dequantizing makes it obvious:
--
--     with scale=0.006, zero_point=-128:
--       int8 -128  ->  (-128 - -128) * 0.006 =  0.000   <-- this is real zero
--       int8    0  ->  (   0 - -128) * 0.006 = +0.768   <-- this is NOT zero
--
-- So clamping at integer 0 would clamp at real +0.768, destroying every
-- genuinely positive activation below that and silently changing the network's
-- output. No crash, no warning -- just wrong answers, which on a drone means a
-- detector that quietly stops seeing things.
--
-- WHY THIS IS COMBINATIONAL (no clock)
-- ------------------------------------
-- ReLU has no state. It is one comparison and one select -- in hardware, a
-- comparator driving a 2:1 multiplexer. It needs no register, so it costs no
-- clock cycle and can sit in the same cycle as whatever feeds it.
--
-- This is a real advantage over the C version: in software ReLU is a loop over
-- N elements costing N iterations. In hardware it is wires, and it is free.

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity relu_int8 is
    generic (
        DATA_WIDTH : positive := 8
    );
    port (
        x          : in  signed(DATA_WIDTH-1 downto 0);   -- quantized input
        zero_point : in  signed(DATA_WIDTH-1 downto 0);   -- the integer meaning real 0.0
        y          : out signed(DATA_WIDTH-1 downto 0)    -- max(zero_point, x)
    );
end entity relu_int8;

architecture rtl of relu_int8 is
begin

    -- One comparator, one mux. Note this compares against zero_point, NOT 0 --
    -- which is the entire lesson of this file.
    y <= x when x > zero_point else zero_point;

end architecture rtl;
