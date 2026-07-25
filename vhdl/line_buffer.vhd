-- =====================================================================
-- line_buffer.vhd — form 3x3 windows from a raster pixel stream
-- =====================================================================
--
-- THE PROBLEM THIS SOLVES
-- -----------------------
-- conv3x3 needs nine pixels at once. A camera gives you ONE pixel at a time,
-- in raster order: row 0 left-to-right, then row 1, and so on. Nothing gives
-- you a 3x3 window.
--
-- The naive fix is to buffer the whole frame and index into it. For an 8x8 test
-- image that is 64 bytes and fine. For a 1920x1080 frame it is 2 MB, which does
-- not fit in the block RAM of a small FPGA -- and you would be paying for
-- memory you do not need.
--
-- THE INSIGHT
-- -----------
-- To form a 3x3 window you never need more than TWO ROWS plus three pixels of
-- history. Everything older has already been used and can be discarded.
--
--     buffer size = 2 * IMG_WIDTH + 3        (19 bytes for an 8-wide image)
--                 NOT IMG_WIDTH * IMG_HEIGHT (64 bytes)
--
-- For 1920-wide video that is 3,843 bytes instead of 2 MB -- a 540x saving, and
-- the reason streaming architectures exist. This is the single most important
-- structural idea in FPGA image processing.
--
-- HOW IT WORKS
-- ------------
-- One long shift register. Every incoming pixel shifts everything along by one.
-- Nine fixed positions are tapped off, and those nine taps ARE the window:
--
--     sr(0)                       <- newest pixel
--     sr(1) sr(2)                    ... one row back ...
--     sr(W) sr(W+1) sr(W+2)          <- W = IMG_WIDTH
--     sr(2W) sr(2W+1) sr(2W+2)    <- oldest pixel in the window
--
--     tap for offset (dy,dx) = sr( (W+1) - dy*W - dx )
--
-- Check the corners: (dy=-1,dx=-1) -> sr(2W+2), the oldest. (dy=+1,dx=+1) ->
-- sr(0), the newest. (0,0) -> sr(W+1), the centre. The window centre trails the
-- input by W+1 pixels, which is the pipeline latency of this block.
--
-- ZERO PADDING IN A STREAM
-- ------------------------
-- At an image edge some taps refer to pixels outside the frame. A frame buffer
-- can just check indices; a stream cannot, because the shift register has no
-- idea where the frame boundaries are. So this block tracks the centre's (y,x)
-- and forces any out-of-range tap to zero.
--
-- Because these vectors use SYMMETRIC quantization (zero_point = 0), integer 0
-- really is real 0.0. With an asymmetric zero point the pad value would have to
-- be zero_point instead -- the same trap as ReLU.
--
-- FLUSHING
-- --------
-- The centre trails the input by W+1, so after the last real pixel the final
-- W+1 windows have not emerged yet. The caller must keep clocking (data value
-- irrelevant) until win_valid has fired IMG_W*IMG_H times. Those trailing
-- pixels never corrupt anything: every tap that could touch them is
-- out-of-frame and therefore zeroed.

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.cnn_types.all;

entity line_buffer is
    generic (
        IMG_W : positive := 8;
        IMG_H : positive := 8
    );
    port (
        clk       : in  std_logic;
        rst       : in  std_logic;
        pix_valid : in  std_logic;                      -- shift a pixel in
        pix_in    : in  signed(7 downto 0);
        window    : out int8_array(0 to 8);             -- row-major 3x3
        win_valid : out std_logic;                      -- window is meaningful
        win_y     : out natural range 0 to IMG_H-1;     -- centre row (verification)
        win_x     : out natural range 0 to IMG_W-1      -- centre col (verification)
    );
end entity line_buffer;

architecture rtl of line_buffer is

    -- Two rows plus three pixels. This is the whole memory cost.
    constant SR_LEN : positive := 2*IMG_W + 3;

    signal sr : int8_array(0 to SR_LEN-1) := (others => (others => '0'));

    -- How many pixels have been shifted in. The centre's raster index is
    -- (count-1) - (IMG_W+1), so this counter is what makes padding possible.
    signal count : natural range 0 to (IMG_W*IMG_H + SR_LEN + 2) := 0;

begin

    -- ---------------------------------------------------------------
    -- The shift register
    -- ---------------------------------------------------------------
    process (clk, rst)
    begin
        if rst = '1' then
            sr    <= (others => (others => '0'));
            count <= 0;
        elsif rising_edge(clk) then
            if pix_valid = '1' then
                -- Shift everything one position older, newest pixel into sr(0).
                -- In hardware this is just wires between flip-flops: no loop
                -- executes at run time, the whole shift happens in one cycle.
                sr(1 to SR_LEN-1) <= sr(0 to SR_LEN-2);
                sr(0)             <= pix_in;
                count             <= count + 1;
            end if;
        end if;
    end process;

    -- ---------------------------------------------------------------
    -- Tap the window, applying zero padding at the frame edges
    -- ---------------------------------------------------------------
    process (sr, count)
        variable centre : integer;                 -- raster index of the centre
        variable cy, cx : integer;
        variable ty, tx : integer;                 -- tap coordinates
        variable idx    : natural;
    begin
        centre := (count - 1) - (IMG_W + 1);

        if centre >= 0 and centre < IMG_W*IMG_H then
            win_valid <= '1';
            cy := centre / IMG_W;
            cx := centre mod IMG_W;
            win_y <= cy;
            win_x <= cx;

            idx := 0;
            for dy in -1 to 1 loop
                for dx in -1 to 1 loop
                    ty := cy + dy;
                    tx := cx + dx;
                    if ty >= 0 and ty < IMG_H and tx >= 0 and tx < IMG_W then
                        -- The tap formula derived in the header comment.
                        window(idx) <= sr((IMG_W + 1) - dy*IMG_W - dx);
                    else
                        window(idx) <= to_signed(0, 8);    -- zero padding
                    end if;
                    idx := idx + 1;
                end loop;
            end loop;
        else
            win_valid <= '0';
            win_y     <= 0;
            win_x     <= 0;
            window    <= (others => (others => '0'));
        end if;
    end process;

end architecture rtl;
