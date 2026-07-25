-- =====================================================================
-- tb_mac_unit.vhd — Testbench for the MAC unit
-- =====================================================================
--
-- WHAT A TESTBENCH IS
-- -------------------
-- Not a program. It is a simulated circuit whose only job is to drive the
-- inputs of another circuit and check its outputs. It has no ports -- it IS
-- the outside world.
--
-- WHAT THIS ONE CHECKS
-- --------------------
--   1. Extremes:      127*127 and -128*-128, the values that overflow int8
--   2. Sign handling: negative operands accumulate correctly
--   3. clr:           clears mid-stream without disturbing the next sum
--   4. THE REAL ONE:  nine MACs reproducing one 3x3 convolution window,
--                     checked against the number NumPy/C/C++ already agree on
--
-- Test 4 is the point. Tests 1-3 confirm the arithmetic; test 4 confirms
-- this hardware computes the same thing as the golden model. That is the
-- claim the whole project is built to support.

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity tb_mac_unit is
end entity tb_mac_unit;

architecture sim of tb_mac_unit is

    constant DATA_WIDTH : positive := 8;
    constant ACC_WIDTH  : positive := 32;
    constant CLK_PERIOD : time     := 10 ns;

    signal clk : std_logic := '0';
    signal rst : std_logic := '1';
    signal clr : std_logic := '0';
    signal en  : std_logic := '0';
    signal a   : signed(DATA_WIDTH-1 downto 0) := (others => '0');
    signal b   : signed(DATA_WIDTH-1 downto 0) := (others => '0');
    signal acc : signed(ACC_WIDTH-1 downto 0);

    signal sim_done : boolean := false;
    signal errors   : natural := 0;

begin

    -- ---------------------------------------------------------------
    -- Clock generator. Stops when sim_done, otherwise the simulation
    -- would never terminate -- a clock is an infinite process.
    -- ---------------------------------------------------------------
    clk_gen : process
    begin
        while not sim_done loop
            clk <= '0'; wait for CLK_PERIOD/2;
            clk <= '1'; wait for CLK_PERIOD/2;
        end loop;
        wait;
    end process;

    -- ---------------------------------------------------------------
    -- Device Under Test
    -- ---------------------------------------------------------------
    dut : entity work.mac_unit
        generic map (DATA_WIDTH => DATA_WIDTH, ACC_WIDTH => ACC_WIDTH)
        port map (clk => clk, rst => rst, clr => clr, en => en,
                  a => a, b => b, acc => acc);

    -- ---------------------------------------------------------------
    -- Stimulus and checking
    -- ---------------------------------------------------------------
    stim : process

        -- Feed one operand pair and accumulate it on the next rising edge.
        procedure mac(av, bv : integer) is
        begin
            a  <= to_signed(av, DATA_WIDTH);
            b  <= to_signed(bv, DATA_WIDTH);
            en <= '1';
            wait until rising_edge(clk);
            en <= '0';
        end procedure;

        procedure clear_acc is
        begin
            clr <= '1';
            wait until rising_edge(clk);
            clr <= '0';
            wait for 1 ns;                      -- let the output settle
        end procedure;

        -- Compare and report. Counting failures rather than aborting on the
        -- first one means a single run tells you everything that is broken.
        procedure check(test_name : string; expected : integer) is
        begin
            wait for 1 ns;
            if to_integer(acc) = expected then
                report "PASS  " & test_name & ": " & integer'image(to_integer(acc));
            else
                report "FAIL  " & test_name &
                       ": got " & integer'image(to_integer(acc)) &
                       ", expected " & integer'image(expected)
                       severity error;
                errors <= errors + 1;
            end if;
        end procedure;

    begin
        -- Release reset
        wait for CLK_PERIOD * 2;
        rst <= '0';
        wait until rising_edge(clk);

        report "=== Test 1: int8 extremes (these overflow int8, which is the point) ===";
        clear_acc;
        mac(127, 127);
        check("127*127", 16129);

        clear_acc;
        mac(-128, -128);
        check("-128*-128", 16384);

        clear_acc;
        mac(-128, 127);
        check("-128*127", -16256);

        report "=== Test 2: accumulation across cycles ===";
        clear_acc;
        mac(100, 100);      -- 10,000
        mac(100, 100);      -- 20,000
        mac(100, 100);      -- 30,000
        check("3 x (100*100)", 30000);

        report "=== Test 3: nine worst-case products need more than 16 bits ===";
        clear_acc;
        for i in 1 to 9 loop
            mac(-128, -128);
        end loop;
        -- 9 * 16,384 = 147,456. A 16-bit accumulator would have wrapped long ago.
        check("9 x (-128*-128)", 147456);

        report "=== Test 4: clr must not leak state into the next sum ===";
        clear_acc;
        mac(50, 50);
        clear_acc;
        mac(2, 3);
        check("clr isolates sums", 6);

        report "=== Test 5: GOLDEN MODEL - one real 3x3 convolution window ===";
        -- Taken from image0 at (y=4,x=4), filter 0, quantized symmetrically.
        -- Input window rows 3,4,5 (int8):     0   0   0
        --                                   127 127 127
        --                                   127 127 127
        -- Kernel (int8):                   -127 -127 -127
        --                                     0    0    0
        --                                   127  127  127
        --
        -- Expected: 3*(127*127) = 48,387  -- and that is exactly the peak the
        -- golden model measured across the whole dataset. This single window IS
        -- the worst case that sizes the accumulator.
        clear_acc;
        mac(0, -127);   mac(0, -127);   mac(0, -127);      -- top row   -> 0
        mac(127, 0);    mac(127, 0);    mac(127, 0);       -- mid row   -> 0
        mac(127, 127);  mac(127, 127);  mac(127, 127);     -- bottom    -> 48,387
        check("conv window (0,4,4) f0", 48387);

        -- ---------------------------------------------------------------
        report "=== SUMMARY ===";
        if errors = 0 then
            report "ALL TESTS PASSED - MAC matches the golden model" severity note;
        else
            report integer'image(errors) & " TEST(S) FAILED" severity failure;
        end if;

        sim_done <= true;
        wait;
    end process;

end architecture sim;
