CXX      := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -I include
CVFLAGS  := $(shell pkg-config --cflags opencv4)
CVLIBS   := $(shell pkg-config --libs opencv4)
GTFLAGS  := -lgtest -lgtest_main -pthread

SRC      := src/image_processing.cpp src/classification.cpp
MAIN_SRC := src/main.cpp
TEST_SRC := test/test_classification.cpp

TARGET   := shape_classifier
TEST_BIN := run_tests

# --- Main build ---
all: $(TARGET)

$(TARGET): $(MAIN_SRC) $(SRC)
	$(CXX) $(CXXFLAGS) $(CVFLAGS) -o $@ $^ $(CVLIBS)

# --- Test build & run ---
test: $(TEST_BIN)
	./$(TEST_BIN)

$(TEST_BIN): $(TEST_SRC) $(SRC)
	$(CXX) $(CXXFLAGS) $(CVFLAGS) -o $@ $^ $(CVLIBS) $(GTFLAGS)

clean:
	rm -f $(TARGET) $(TEST_BIN)

.PHONY: all test clean
