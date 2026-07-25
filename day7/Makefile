CC       = gcc
CXX      = g++
CFLAGS   = -Wall -Wextra -O2 -std=c11
CXXFLAGS = -Wall -Wextra -O2 -std=c++17
ASAN     = -O1 -g -fsanitize=address,undefined

all: golden_c golden_cpp

golden_c: golden_c.c model_io.h
	$(CC) $(CFLAGS) -o $@ golden_c.c -lm

golden_cpp: golden_cpp.cpp model_io.h
	$(CXX) $(CXXFLAGS) -o $@ golden_cpp.cpp

# Regenerate the golden model and reference vectors from the NumPy source.
model: export_weights.py
	python3 export_weights.py

# Build everything, then prove NumPy = C = C++.
verify: all
	@test -f weights.bin || python3 export_weights.py
	python3 verify.py

# The loader mallocs the whole file and layers borrow into it; prove nothing
# leaks and no read runs off the end of the buffer.
asan: golden_c.c golden_cpp.cpp model_io.h
	$(CC)  $(CFLAGS)   $(ASAN) -o golden_c_asan   golden_c.c -lm
	$(CXX) $(CXXFLAGS) $(ASAN) -o golden_cpp_asan golden_cpp.cpp
	@test -f weights.bin || python3 export_weights.py
	@echo "--- golden_c_asan ---"
	@./golden_c_asan   weights.bin > /dev/null && echo "    clean"
	@echo "--- golden_cpp_asan ---"
	@./golden_cpp_asan weights.bin > /dev/null && echo "    clean"

# Show the loader rejecting a corrupt file instead of trusting it.
test-robust: golden_c
	@python3 -c "import pathlib; b=bytearray(pathlib.Path('weights.bin').read_bytes()); b[100]^=0xFF; pathlib.Path('corrupt.bin').write_bytes(b)"
	@echo "--- feeding a corrupted model (one flipped byte) ---"
	@./golden_c corrupt.bin > /dev/null; echo "    exit code: $$? (non-zero = correctly rejected)"
	@rm -f corrupt.bin

clean:
	rm -f golden_c golden_cpp golden_c_asan golden_cpp_asan corrupt.bin

.PHONY: all model verify asan test-robust clean
