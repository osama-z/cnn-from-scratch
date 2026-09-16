CXX      = g++
CXXFLAGS = -Wall -Wextra -O2 -std=c++17

# Sanitizer build: -O1 keeps stack traces readable, -g adds line numbers.
ASANFLAGS = -Wall -Wextra -O1 -g -std=c++17 -fsanitize=address,undefined

all: cnn_forward templated_cnn

# Part 1 — object-oriented CNN: classes, inheritance, unique_ptr, RAII
cnn_forward: cnn_forward.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

# Part 2 — templated CNN: one source, float and int8 instantiations
templated_cnn: templated_cnn.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

# Prove the RAII claim instead of asserting it.
# Both files allocate constantly and contain zero free() and zero delete;
# LeakSanitizer should still report nothing at exit.
asan: cnn_forward.cpp templated_cnn.cpp
	$(CXX) $(ASANFLAGS) -o cnn_forward_asan   cnn_forward.cpp
	$(CXX) $(ASANFLAGS) -o templated_cnn_asan templated_cnn.cpp
	@echo "--- running cnn_forward_asan ---"
	@./cnn_forward_asan   > /dev/null && echo "    clean: no leaks, no UB"
	@echo "--- running templated_cnn_asan ---"
	@./templated_cnn_asan > /dev/null && echo "    clean: no leaks, no UB"

# Show that `if constexpr` really removes the other branch.
# The float instantiation should contain mulss and no imull;
# the int8 instantiation should contain movsbl/imull and no float ops.
asm: templated_cnn.cpp
	$(CXX) $(CXXFLAGS) -S -o templated_cnn.s $<
	@echo "wrote templated_cnn.s -- inspect Conv2D<float> vs Conv2D<signed char>"

run: all
	./cnn_forward
	@echo ""
	./templated_cnn

clean:
	rm -f cnn_forward templated_cnn \
	      cnn_forward_asan templated_cnn_asan \
	      templated_cnn.s asan_report.txt

.PHONY: all asan asm run clean
