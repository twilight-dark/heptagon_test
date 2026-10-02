CXX      ?= g++
CXXFLAGS ?= -O3 -std=c++17 -fopenmp -Wall -Wextra -Iinc
LDFLAGS  ?= -pthread

NATIVE ?= 1
ifeq ($(NATIVE),1)
  CXXFLAGS += -march=native
endif

TARGET := cpu_pointer_chasing
SRCS   := src/main.cpp src/chasing.cpp src/utility.cpp
HDRS   := inc/utility.h
OBJS   := $(SRCS:.cpp=.o)

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

%.o: %.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

run: $(TARGET)
	./$(TARGET) gen -c 4M -s 1 -o my_table.bin
	./$(TARGET) bench -i my_table.bin

clean:
	rm -f $(OBJS) $(TARGET) my_table.bin
