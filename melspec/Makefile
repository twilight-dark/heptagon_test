CXX_COMPLIER = g++
CXX_FLAGS = -std=c++17 -Wall -Wextra -O2
LINKER = g++
LINKER_FLAGS = 
INCLUDES = -I./include

SRCS = $(wildcard src/*.cpp)
INCS = $(wildcard include/*.h)
OBJS = $(SRCS:.cpp=.o)
EXEC = melspec

.PHONY: all clean test format

all: $(EXEC)

$(EXEC): $(OBJS)
	$(LINKER) $(LINKER_FLAGS) -o $@ $^

%.o: %.cpp
	$(CXX_COMPLIER) $(CXX_FLAGS) $(INCLUDES) -o $@ -c $<

clean:
	rm -f $(OBJS) $(EXEC)

test: $(EXEC)
	python3 ./validate.py ./$(EXEC)

format: $(SRCS) $(INCS)
	clang-format -i $^
