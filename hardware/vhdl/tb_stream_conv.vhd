-- =====================================================================
-- tb_stream_conv.vhd — line_buffer + conv3x3, streaming, vs the golden model
-- =====================================================================
--
-- This is the full streaming datapath: pixels go in one at a time, windows come
-- out of line_buffer, conv3x3 convolves them, and every result is checked
-- against lessons/day7/vhdl_vectors/.
--
--     pix_in --> [ line_buffer ] --window--> [ conv3x3 ] --acc--> check
--
-- Same 384 comparisons as tb_conv3x3, but now the windows are FORMED IN
-- HARDWARE from a raster stream rather than handed over pre-built by the
-- testbench. That is the difference between a proven arithmetic unit and a
-- proven datapath.
--
-- WHY THIS IS A STRONGER RESULT
-- -----------------------------
-- tb_conv3x3 proved the arithmetic. It said nothing about whether windows can
-- actually be assembled from a camera-style stream, which is where the real
-- bugs live: off-by-one taps, wrong latency, edges handled wrongly, state
-- leaking between rows. Getting 384/384 through the line buffer means the
-- addressing is right too.

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;
use work.cnn_types.all;

entity tb_stream_conv is
end entity tb_stream_conv;

architecture sim of tb_stream_conv is

    constant IMG_W     : positive := 8;
    constant IMG_H     : positive := 8;
    constant N_FILTERS : natural  := 2;
    constant VEC_DIR   : string   := "../../lessons/day7/vhdl_vectors/";
    constant CLK_PER   : time     := 10 ns;

    signal clk       : std_logic := '0';
    signal rst       : std_logic := '1';
    signal pix_valid : std_logic := '0';
    signal pix_in    : signed(7 downto 0) := (others => '0');

    signal window    : int8_array(0 to 8);
    signal win_valid : std_logic;
    signal win_y     : natural range 0 to IMG_H-1;
    signal win_x     : natural range 0 to IMG_W-1;

    signal kernel : int8_array(0 to 8) := (others => (others => '0'));
    signal acc    : signed(31 downto 0);

    signal sim_done : boolean := false;
    signal errors   : natural := 0;
    signal checked  : natural := 0;

begin

    clk_gen : process
    begin
        while not sim_done loop
            clk <= '0'; wait for CLK_PER/2;
            clk <= '1'; wait for CLK_PER/2;
        end loop;
        wait;
    end process;

    -- The datapath under test: window formation feeding the convolver.
    lb : entity work.line_buffer
        generic map (IMG_W => IMG_W, IMG_H => IMG_H)
        port map (clk => clk, rst => rst,
                  pix_valid => pix_valid, pix_in => pix_in,
                  window => window, win_valid => win_valid,
                  win_y => win_y, win_x => win_x);

    cv : entity work.conv3x3
        port map (window => window, kernel => kernel, acc => acc);

    stim : process

        type int_vector is array (natural range <>) of integer;
        variable img_data : int_vector(0 to IMG_W*IMG_H-1);
        variable weights  : int_vector(0 to N_FILTERS*9-1);
        variable expected : int_vector(0 to N_FILTERS*IMG_W*IMG_H-1);

        procedure read_ints(fname : in string; dst : out int_vector; n : in natural) is
            file     f  : text;
            variable l  : line;
            variable v  : integer;
            variable i  : natural := 0;
            variable ok : boolean;
        begin
            file_open(f, fname, read_mode);
            while i < n and not endfile(f) loop
                readline(f, l);
                if l.all'length > 0 and l.all(l.all'left) /= '#' then
                    read(l, v, ok);
                    if ok then dst(i) := v; i := i + 1; end if;
                end if;
            end loop;
            file_close(f);
            assert i = n report "read_ints: " & fname & " short" severity failure;
        end procedure;

        variable exp_val : integer;
        variable seen    : natural;
    begin
        report "=== STREAMING DATAPATH vs GOLDEN MODEL ===";
        report "    line_buffer forms windows from a raster stream, conv3x3 convolves";
        report "    buffer size = 2*W+3 = " & integer'image(2*IMG_W+3) &
               " bytes, NOT W*H = " & integer'image(IMG_W*IMG_H);

        read_ints(VEC_DIR & "conv_weights_int8.txt", weights, N_FILTERS*9);

        for n in 0 to 2 loop
            read_ints(VEC_DIR & "image" & integer'image(n) & "_int8.txt",
                      img_data, IMG_W*IMG_H);
            read_ints(VEC_DIR & "image" & integer'image(n) & "_conv_acc_int32.txt",
                      expected, N_FILTERS*IMG_W*IMG_H);

            for f in 0 to N_FILTERS-1 loop
                -- Load this filter's kernel, then reset and re-stream the image.
                for i in 0 to 8 loop
                    kernel(i) <= to_signed(weights(f*9 + i), 8);
                end loop;

                rst <= '1';
                wait until rising_edge(clk);
                rst <= '0';
                wait until rising_edge(clk);

                seen := 0;

                -- Stream the image, then keep clocking to flush the pipeline.
                -- The centre trails the input by W+1, so the last W+2 windows
                -- only emerge after the real pixels have run out.
                for p in 0 to IMG_W*IMG_H + IMG_W + 2 loop
                    if p < IMG_W*IMG_H then
                        pix_in <= to_signed(img_data(p), 8);
                    else
                        pix_in <= to_signed(0, 8);    -- flush; never reaches a
                    end if;                           -- valid tap (all out-of-frame)
                    pix_valid <= '1';
                    wait until rising_edge(clk);
                    pix_valid <= '0';
                    wait for 1 ns;                    -- combinational settle

                    if win_valid = '1' then
                        exp_val := expected(f*IMG_W*IMG_H + win_y*IMG_W + win_x);
                        checked <= checked + 1;
                        seen    := seen + 1;

                        if to_integer(acc) /= exp_val then
                            report "FAIL image " & integer'image(n) &
                                   " filter " & integer'image(f) &
                                   " (y=" & integer'image(win_y) &
                                   ",x=" & integer'image(win_x) & "): got " &
                                   integer'image(to_integer(acc)) &
                                   ", expected " & integer'image(exp_val)
                                severity error;
                            errors <= errors + 1;
                            wait for 1 ns;
                        end if;
                    end if;
                end loop;

                -- Every pixel must have produced exactly one window. If the
                -- latency or flush count were wrong this would catch it even
                -- when the values that DID emerge were all correct.
                assert seen = IMG_W*IMG_H
                    report "image " & integer'image(n) & " filter " &
                           integer'image(f) & ": got " & integer'image(seen) &
                           " windows, expected " & integer'image(IMG_W*IMG_H)
                    severity error;
            end loop;

            report "    image " & integer'image(n) & " streamed";
        end loop;

        wait for 1 ns;
        report "=== SUMMARY ===";
        report "    comparisons: " & integer'image(checked);
        if errors = 0 then
            report "ALL 384 STREAMED OUTPUTS MATCH - full datapath == NumPy == C == C++"
                severity note;
        else
            report integer'image(errors) & " MISMATCH(ES)" severity failure;
        end if;

        sim_done <= true;
        wait;
    end process;

end architecture sim;
