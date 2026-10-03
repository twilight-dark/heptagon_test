NVCC := nvcc

CUDA_ARCH := -arch=sm_89

NVCCFLAGS := -std=c++17 -O3 $(CUDA_ARCH) --use_fast_math \
    -maxrregcount=128 \
    --extra-device-vectorization \
    -Xcompiler="-O3,-march=native,-mtune=native"

INCLUDES := -Iinclude

TARGET := bench_mark
OBJECTS := src/main.o src/trans.o

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(NVCC) $(CUDA_ARCH) $(OBJECTS) -o $(TARGET)

src/main.o: src/main.cpp include/trans.h
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

src/trans.o: src/trans.cu include/trans.h
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

run: $(TARGET)
	./$(TARGET) data/cases.txt

clean:
	rm -f $(TARGET) $(OBJECTS)

.PHONY: all run clean