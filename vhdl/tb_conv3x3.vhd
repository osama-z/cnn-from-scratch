-- =====================================================================
-- tb_conv3x3.vhd — GOLDEN MODEL VERIFICATION
-- =====================================================================
--
-- THIS IS THE FILE THE WHOLE PROJECT WAS BUILT TO MAKE POSSIBLE.
--
-- It reads the exact files that day7/export_weights.py wrote:
--
--     conv_weights_int8.txt        18 values  (2 filters x 3x3)
--     image{0,1,2}_int8.txt        64 values  (8x8 quantized image)
--     image{N}_conv_acc_int32.txt 128 values  (2 filters x 8x8 expected acc)
--
-- and checks all 128 convolution outputs per image against them. Three images
-- means 384 independent comparisons between this hardware and the NumPy
-- reference that C and C++ already match to 1.2e-10.
--
-- WHY READ FILES INSTEAD OF HARDCODING VALUES
-- -------------------------------------------
-- Hardcoded expected values drift the moment the model changes, and they encode
-- the author's belief about what is correct rather than what the reference
-- actually produced. Reading the reference's own output files means:
--
--   * regenerating the model automatically regenerates the test
--   * the hardware is checked against software, not against a comment
--   * there is exactly ONE definition of correct in the whole project
--
-- This is standard practice in hardware verification: RTL is checked against a
-- C or Python reference model, not against hand-computed tables.
--
-- A VHDL GOTCHA WORTH KNOWING
-- ---------------------------
-- VHDL is CASE-INSENSITIVE. `img` and `IMG` are the same identifier, so a
-- variable named `img` silently shadows a constant named `IMG` and every later
-- use of `IMG-1` becomes "subtract 1 from an array" -- which fails with the
-- baffling "no function declarations for operator -". Hence `img_data` below.
-- Coming from C or Python, this is the easiest VHDL trap to fall into.
--
-- ZERO PADDING
-- ------------
-- Pixels outside the image contribute nothing. Because these vectors use
-- SYMMETRIC quantization (zero_point = 0), integer 0 really does mean real 0.0,
-- so padding with integer 0 is correct. With an asymmetric zero_point it would
-- have to pad with zero_point instead -- the same trap as ReLU.

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;
use work.cnn_types.all;

entity tb_conv3x3 is
end entity tb_conv3x3;

architecture sim of tb_conv3x3 is

    constant IMG       : natural := 8;     -- 8x8 images
    constant N_FILTERS : natural := 2;
    constant VEC_DIR   : string  := "../day7/vhdl_vectors/";

    signal window : int8_array(0 to 8) := (others => (others => '0'));
    signal kernel : int8_array(0 to 8) := (others => (others => '0'));
    signal acc    : signed(31 downto 0);

    signal errors  : natural := 0;
    signal checked : natural := 0;

begin

    dut : entity work.conv3x3
        port map (window => window, kernel => kernel, acc => acc);

    stim : process

        -- Storage for one image, its weights, and the expected accumulators.
        type int_vector is array (natural range <>) of integer;
        variable img_data : int_vector(0 to IMG*IMG-1);
        variable weights  : int_vector(0 to N_FILTERS*9-1);
        variable expected : int_vector(0 to N_FILTERS*IMG*IMG-1);

        -- Read `n` integers, one per line, skipping blank and '#' comment lines.
        procedure read_ints(fname : in string;
                            dst   : out int_vector;
                            n     : in  natural) is
            file     f    : text;
            variable l    : line;
            variable v    : integer;
            variable i    : natural := 0;
            variable ok   : boolean;
        begin
            file_open(f, fname, read_mode);
            while i < n and not endfile(f) loop
                readline(f, l);
                if l.all'length > 0 and l.all(l.all'left) /= '#' then
                    read(l, v, ok);
                    if ok then
                        dst(i) := v;
                        i := i + 1;
                    end if;
                end if;
            end loop;
            file_close(f);
            assert i = n
                report "read_ints: " & fname & " gave " & integer'image(i) &
                       " values, expected " & integer'image(n)
                severity failure;
        end procedure;

        -- Build the 3x3 window centred on (y,x), padding outside with 0.
        procedure set_window(pix : in int_vector; y, x : in integer) is
            variable iy, ix : integer;
            variable idx    : natural;
        begin
            idx := 0;
            for ky in -1 to 1 loop
                for kx in -1 to 1 loop
                    iy := y + ky;
                    ix := x + kx;
                    if iy >= 0 and iy < IMG and ix >= 0 and ix < IMG then
                        window(idx) <= to_signed(pix(iy*IMG + ix), 8);
                    else
                        window(idx) <= to_signed(0, 8);      -- zero padding
                    end if;
                    idx := idx + 1;
                end loop;
            end loop;
        end procedure;

        procedure set_kernel(w : in int_vector; filter : in natural) is
        begin
            for i in 0 to 8 loop
                kernel(i) <= to_signed(w(filter*9 + i), 8);
            end loop;
        end procedure;

        variable exp_val   : integer;

    begin
        report "=== GOLDEN MODEL VERIFICATION: conv3x3 vs NumPy reference ===";
        report "    reading vectors from " & VEC_DIR;

        read_ints(VEC_DIR & "conv_weights_int8.txt", weights, N_FILTERS*9);
        report "    loaded " & integer'image(N_FILTERS*9) & " int8 weights";

        for n in 0 to 2 loop
            read_ints(VEC_DIR & "image" & integer'image(n) & "_int8.txt",
                      img_data, IMG*IMG);
            read_ints(VEC_DIR & "image" & integer'image(n) & "_conv_acc_int32.txt",
                      expected, N_FILTERS*IMG*IMG);

            report "--- image " & integer'image(n) & ": checking " &
                   integer'image(N_FILTERS*IMG*IMG) & " outputs ---";

            for f in 0 to N_FILTERS-1 loop
                set_kernel(weights, f);
                for y in 0 to IMG-1 loop
                    for x in 0 to IMG-1 loop
                        set_window(img_data, y, x);
                        wait for 1 ns;               -- let combinational logic settle

                        exp_val := expected(f*IMG*IMG + y*IMG + x);
                        checked <= checked + 1;

                        if to_integer(acc) /= exp_val then
                            report "FAIL image " & integer'image(n) &
                                   " filter " & integer'image(f) &
                                   " (y=" & integer'image(y) &
                                   ",x=" & integer'image(x) & "): got " &
                                   integer'image(to_integer(acc)) &
                                   ", expected " & integer'image(exp_val)
                                severity error;
                            errors <= errors + 1;
                            wait for 1 ns;
                        end if;
                    end loop;
                end loop;
            end loop;

            report "    image " & integer'image(n) & " done";
        end loop;

        wait for 1 ns;
        report "=== SUMMARY ===";
        report "    comparisons: " & integer'image(checked);
        if errors = 0 then
            report "ALL 384 OUTPUTS MATCH - VHDL conv3x3 == NumPy == C == C++"
                severity note;
        else
            report integer'image(errors) & " MISMATCH(ES)" severity failure;
        end if;
        wait;
    end process;

end architecture sim;
