# Makefile for CUDA build
# Usage:
#   make
#   make run numElements=32768 threadsPerBlock=256
#   make clean

NVCC ?= nvcc
TARGET := slow
SRC := slow.cu
NVCCFLAGS ?= -O3 -std=c++20 --ptxas-options=-v

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

run: $(TARGET)
	./$(TARGET) $(numElements) $(threadsPerBlock)

clean:
	rm -f $(TARGET)