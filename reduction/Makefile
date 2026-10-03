NVCC := nvcc

CUDA_ARCH := -arch=sm_89
NVCCFLAGS := -std=c++17 -O3 $(CUDA_ARCH) --use_fast_math \
    -maxrregcount=128 \
    -Xcompiler="-O3,-march=native,-mtune=native"

INCLUDES := -Iinclude -Isrc
TARGET := reduction
OBJECTS := src/main.o src/reduction.o src/baseline.o

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(NVCC) $(CUDA_ARCH) $(OBJECTS) -o $@

src/main.o: src/main.cpp include/reduction.h
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -x cu -c $< -o $@

src/reduction.o: src/reduction.cu include/reduction.h
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

src/baseline.o: src/baseline.cu src/baseline.h
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

run: $(TARGET)
	./$(TARGET) data/cases.txt

clean:
	rm -f $(TARGET) $(OBJECTS) std/*.o

.PHONY: all run clean
