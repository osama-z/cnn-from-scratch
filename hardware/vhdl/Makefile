# =====================================================================
# GHDL build for the Phase 1 hardware
# =====================================================================
#
# GHDL runs in three steps, which mirror how a real toolchain works:
#   -a  ANALYZE    parse and type-check one file into the work library
#   -e  ELABORATE  link an entity and its children into a simulatable design
#   -r  RUN        simulate it
#
# --std=08 selects VHDL-2008. -fsynopsys is NOT used: it enables the
# non-standard std_logic_arith library, and relying on it teaches habits that
# break on real synthesis tools. numeric_std is the standard, portable choice.

GHDL      = ghdl
STD       = --std=08
WORKDIR   = build
GHDLFLAGS = $(STD) --workdir=$(WORKDIR)

SRC = cnn_types.vhd mac_unit.vhd relu_int8.vhd requantize.vhd conv3x3.vhd line_buffer.vhd
TB  = tb_mac_unit.vhd tb_relu_requant.vhd tb_conv3x3.vhd tb_stream_conv.vhd

all: test

$(WORKDIR):
	@mkdir -p $(WORKDIR)

# Analyze every source file. -Wall surfaces the warnings that matter.
analyze: | $(WORKDIR)
	$(GHDL) -a $(GHDLFLAGS) $(SRC) $(TB)

# Elaborate + run. --assert-level=error makes a failed `report ... severity
# failure` stop the simulation with a non-zero exit code, so `make test`
# actually fails in CI rather than printing errors and returning success.
test: analyze
	@echo "--- tb_mac_unit ---"
	@$(GHDL) --elab-run $(GHDLFLAGS) tb_mac_unit --assert-level=error
	@echo "--- tb_relu_requant ---"
	@$(GHDL) --elab-run $(GHDLFLAGS) tb_relu_requant --assert-level=error
	@echo "--- tb_conv3x3 (golden model, windows given) ---"
	@$(GHDL) --elab-run $(GHDLFLAGS) tb_conv3x3 --assert-level=error
	@echo "--- tb_stream_conv (golden model, windows formed in HW) ---"
	@$(GHDL) --elab-run $(GHDLFLAGS) tb_stream_conv --assert-level=error

# Same, but dump waveforms for GTKWave. Seeing the accumulator climb across
# nine cycles is the fastest way to understand what a MAC actually does.
wave: analyze
	$(GHDL) --elab-run $(GHDLFLAGS) tb_mac_unit --wave=$(WORKDIR)/mac.ghw
	@echo "open with:  gtkwave $(WORKDIR)/mac.ghw"

# Check for synthesisability. GHDL's synth pass rejects constructs that
# simulate fine but cannot become hardware -- a genuinely useful gate before
# taking anything near Vivado.
synth: analyze
	@set -e; for u in mac_unit relu_int8 requantize conv3x3 line_buffer; do \
	  $(GHDL) --synth $(GHDLFLAGS) $$u > $(WORKDIR)/$${u}_synth.vhd; \
	  echo "  synthesisable: $$u"; \
	done

clean:
	rm -rf $(WORKDIR)

.PHONY: all analyze test wave synth clean
