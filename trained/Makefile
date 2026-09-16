CC = gcc
CXX = g++
PYTHON ?= python3
CPPFLAGS = -D_POSIX_C_SOURCE=200809L
CFLAGS = -O2 -Wall -Wextra -std=c11
CXXFLAGS = -O2 -Wall -Wextra -std=c++17
ASAN = -O1 -g -fsanitize=address,undefined
HEADERS = run_common.h ../engine/model_io.h ../engine/cnn_c.h ../engine/cnn.hpp

all: infer_c infer_cpp

infer_c: infer_c.c $(HEADERS)
	$(CC) $(CPPFLAGS) $(CFLAGS) -o $@ $< -lm

infer_cpp: infer_cpp.cpp $(HEADERS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -o $@ $<

test: all
	cd .. && $(PYTHON) -m unittest discover -s tests -v

asan: $(HEADERS) infer_c.c infer_cpp.cpp
	$(CC) $(CPPFLAGS) $(CFLAGS) $(ASAN) -o infer_c_asan infer_c.c -lm
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(ASAN) -o infer_cpp_asan infer_cpp.cpp
	cd .. && CNN_TEST_SANITIZERS=1 $(PYTHON) -m unittest discover -s tests -v

.PHONY: all test asan
