CXX      ?= g++
CXXFLAGS ?= -O3 -std=c++17 -march=native -fopenmp -pthread -Wall -Wextra
TARGET   := histogram_bench
SRCS     := src/main.cpp src/histogram.cpp baseline.cpp
OBJS     := $(SRCS:.cpp=.o)

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

src/main.o:     src/main.cpp     baseline.h
src/histogram.o: src/histogram.cpp
baseline.o:     baseline.cpp     baseline.h

run: all
	./$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)