CC       = gcc
CXX      = g++
CFLAGS   = -Wall -Wextra -O2 -std=c11
CXXFLAGS = -Wall -Wextra -O2 -std=c++17
ASAN     = -O1 -g -fsanitize=address,undefined
PYTHON  ?= python3

all: golden_c golden_cpp

golden_c: golden_c.c ../../engine/model_io.h ../../engine/cnn_c.h
	$(CC) $(CFLAGS) -o $@ golden_c.c -lm

golden_cpp: golden_cpp.cpp ../../engine/model_io.h ../../engine/cnn.hpp
	$(CXX) $(CXXFLAGS) -o $@ golden_cpp.cpp

# Regenerate the golden model and reference vectors from the NumPy source.
model: export_weights.py ../../engine/model_format.py
	cd ../.. && $(PYTHON) -m lessons.day7.export_weights

# Build everything, then prove NumPy = C = C++.
verify: all
	@test -f weights.bin || $(MAKE) model PYTHON=$(PYTHON)
	$(PYTHON) verify.py

# The loader mallocs the whole file and layers borrow into it; prove nothing
# leaks and no read runs off the end of the buffer.
asan: golden_c.c golden_cpp.cpp ../../engine/model_io.h ../../engine/cnn.hpp ../../engine/cnn_c.h
	$(CC)  $(CFLAGS)   $(ASAN) -o golden_c_asan   golden_c.c -lm
	$(CXX) $(CXXFLAGS) $(ASAN) -o golden_cpp_asan golden_cpp.cpp
	@test -f weights.bin || $(MAKE) model PYTHON=$(PYTHON)
	@echo "--- golden_c_asan ---"
	@./golden_c_asan   weights.bin > /dev/null && echo "    clean"
	@echo "--- golden_cpp_asan ---"
	@./golden_cpp_asan weights.bin > /dev/null && echo "    clean"
	$(PYTHON) test_robust.py --binaries golden_c_asan golden_cpp_asan

# Show the loader rejecting a corrupt file instead of trusting it.
test-robust: all
	$(PYTHON) test_robust.py

test_engine: test_engine.cpp ../../engine/cnn.hpp
	$(CXX) $(CXXFLAGS) -o $@ $<

test-engine: test_engine
	./test_engine

clean:
	rm -f golden_c golden_cpp golden_c_asan golden_cpp_asan corrupt.bin test_engine

.PHONY: all model verify asan test-robust test-engine clean
