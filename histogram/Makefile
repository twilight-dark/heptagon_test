CXX      ?= g++
CXXFLAGS ?= -O3 -std=c++17 -march=native -fopenmp -pthread -Wall -Wextra
TARGET   := histogram_bench
SRCS     := src/main.cpp src/histogram.cpp baseline.cpp
OBJS     := $(SRCS:.cpp=.o)

.PHONY: all clean run test

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

src/main.o:     src/main.cpp     baseline.h src/histogram.h
src/histogram.o: src/histogram.cpp src/histogram.h
baseline.o:     baseline.cpp     baseline.h

histogram_test: tests/histogram_test.cpp src/histogram.cpp src/histogram.h baseline.cpp baseline.h
	$(CXX) $(CXXFLAGS) -o $@ tests/histogram_test.cpp src/histogram.cpp baseline.cpp

test: histogram_test
	./histogram_test

run: all
	./$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET) histogram_test
