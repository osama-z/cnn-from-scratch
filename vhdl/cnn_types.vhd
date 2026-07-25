-- =====================================================================
-- cnn_types.vhd — shared array types
-- =====================================================================
--
-- VHDL has no built-in "array of signed", so it has to be declared once and
-- shared. Putting it in a package is the VHDL equivalent of a header file.
--
-- The `natural range <>` makes the array UNCONSTRAINED: the type says "an
-- array of signed(7 downto 0)" without fixing the length. Each port or signal
-- that uses it picks its own size. That is how one conv3x3 entity can later be
-- reused for a 5x5 kernel without editing the type.

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

package cnn_types is

    -- A run of int8 values: activations, weights, a flattened window.
    type int8_array is array (natural range <>) of signed(7 downto 0);

    -- A run of int32 values: accumulators.
    type int32_array is array (natural range <>) of signed(31 downto 0);

end package cnn_types;
