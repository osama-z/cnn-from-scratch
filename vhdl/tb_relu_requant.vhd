-- =====================================================================
-- tb_relu_requant.vhd — Testbench for relu_int8 and requantize
-- =====================================================================
--
-- Both units are combinational, so there is no clock here. Apply inputs,
-- wait for the signals to settle, check outputs.
--
-- THE TEST THAT MATTERS MOST is Test 2: it proves that clamping at integer 0
-- instead of at zero_point would corrupt the data. That is Day 5's trap,
-- demonstrated rather than described.

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity tb_relu_requant is
end entity tb_relu_requant;

architecture sim of tb_relu_requant is

    -- ReLU signals
    signal r_x, r_zp, r_y : signed(7 downto 0) := (others => '0');

    -- Requantize signals
    signal q_acc   : signed(31 downto 0) := (others => '0');
    signal q_mult  : signed(31 downto 0) := to_signed(172, 32);   -- M0
    signal q_shift : natural range 0 to 63 := 16;                 -- SHIFT
    signal q_zp    : signed(7 downto 0)  := (others => '0');
    signal q_y     : signed(7 downto 0);

    signal errors : natural := 0;

begin

    dut_relu : entity work.relu_int8
        port map (x => r_x, zero_point => r_zp, y => r_y);

    dut_requant : entity work.requantize
        port map (acc => q_acc, multiplier => q_mult, shift => q_shift,
                  zero_point => q_zp, y => q_y);

    stim : process

        procedure check(test_name : string; got, expected : integer) is
        begin
            if got = expected then
                report "PASS  " & test_name & ": " & integer'image(got);
            else
                report "FAIL  " & test_name &
                       ": got " & integer'image(got) &
                       ", expected " & integer'image(expected)
                       severity error;
                errors <= errors + 1;
            end if;
        end procedure;

        procedure do_relu(xv, zpv : integer) is
        begin
            r_x  <= to_signed(xv, 8);
            r_zp <= to_signed(zpv, 8);
            wait for 1 ns;
        end procedure;

        procedure do_requant(accv : integer) is
        begin
            q_acc <= to_signed(accv, 32);
            wait for 1 ns;
        end procedure;

    begin
        report "=== Test 1: ReLU with zero_point = 0 (symmetric case) ===";
        do_relu(50, 0);    check("relu(50, zp=0)", to_integer(r_y), 50);
        do_relu(-50, 0);   check("relu(-50, zp=0)", to_integer(r_y), 0);
        do_relu(0, 0);     check("relu(0, zp=0)", to_integer(r_y), 0);
        do_relu(127, 0);   check("relu(127, zp=0)", to_integer(r_y), 127);

        report "=== Test 2: DAY 5'S TRAP - zero_point = -128 ===";
        -- With zero_point = -128, the integer -128 represents real 0.0.
        -- Every value above -128 is a genuinely POSITIVE activation and must
        -- pass through untouched.
        --
        -- If this were implemented as max(0, x), then relu(-50) would return 0,
        -- which dequantizes to a LARGE POSITIVE number instead of leaving -50
        -- alone. The bug does not crash; it silently rewrites the feature map.
        do_relu(-50, -128);
        check("relu(-50, zp=-128) passes through", to_integer(r_y), -50);
        do_relu(-128, -128);
        check("relu(-128, zp=-128) = real zero", to_integer(r_y), -128);
        do_relu(-100, -128);
        check("relu(-100, zp=-128) passes through", to_integer(r_y), -100);
        do_relu(100, -128);
        check("relu(100, zp=-128) passes through", to_integer(r_y), 100);

        report "=== Test 3: requantize, M0=172 shift=16 (the real parameters) ===";
        -- M = 0.002624671916, approximated as 172/65536 = 0.002624511719
        do_requant(0);       check("requant(0)", to_integer(q_y), 0);
        do_requant(48387);   check("requant(48387) = PEAK -> int8 max", to_integer(q_y), 127);
        do_requant(30000);   check("requant(30000)", to_integer(q_y), 79);
        do_requant(-48387);  check("requant(-48387)", to_integer(q_y), -127);
        do_requant(127);     check("requant(127) rounds to 0", to_integer(q_y), 0);

        report "=== Test 4: SATURATION must clamp, never wrap ===";
        -- 1,000,000 * M = 2624.67, far past int8. It must clamp to 127.
        -- If it wrapped, 2625 mod 256 = 65 -- a plausible-looking wrong answer,
        -- which is worse than an obviously wrong one.
        do_requant(1000000);
        check("requant(1e6) saturates high", to_integer(q_y), 127);
        do_requant(-1000000);
        check("requant(-1e6) saturates low", to_integer(q_y), -128);

        report "=== Test 5: requantize honours a non-zero output zero_point ===";
        q_zp <= to_signed(-128, 8);
        do_requant(30000);
        -- 79 + (-128) = -49
        check("requant(30000, zp=-128)", to_integer(q_y), -49);
        q_zp <= to_signed(0, 8);
        wait for 1 ns;

        report "=== SUMMARY ===";
        if errors = 0 then
            report "ALL TESTS PASSED - relu and requantize match the C reference"
                severity note;
        else
            report integer'image(errors) & " TEST(S) FAILED" severity failure;
        end if;
        wait;
    end process;

end architecture sim;
