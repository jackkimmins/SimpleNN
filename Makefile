CXX = g++
CXXFLAGS = -std=c++20 -O2

TARGET = bin/main

# Automatically find all .cpp files in the src directory
SOURCES = $(wildcard src/*.cpp)

# Convert the list of source files to a list of object files
OBJECTS = $(SOURCES:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) $(OBJECTS) src/*.o

.PHONY: all clean run
