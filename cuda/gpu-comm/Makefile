CXX := g++
NVCC := nvcc

SRCDIR := src
BUILDDIR := build
LIBDIR := lib
TARGET := bin/tool

SOURCES := $(shell find $(SRCDIR) -type f ) # All source files
CPP_OBJECTS := $(SOURCES:.cpp=.o) # Substitute .cpp for .o
CPP_CU_OBJECTS := $(CPP_OBJECTS:.cu=.o) # Substitute .cu for .o
OBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(CPP_CU_OBJECTS)) # substitute src for build

CXXFLAGS := -std=c++11 -Wall -Wextra -Wshadow -Wpedantic
NVCCFLAGS := -std=c++11 -Xcompiler=-Wall,-Wextra,-Wshadow -arch=sm_35

LIB := -Llib -L/usr/local/cuda/lib64 -lcudart
INC := -Iinclude

all: $(TARGET)

# Use CXX for the final linking. Have to tell it where cuda is
$(TARGET): $(OBJECTS)
	$(CXX) $^ -o $(TARGET) $(LIB)

$(TARGET_LIB): $(LIB_OBJECTS)
	@mkdir -p $(LIBDIR)
	$(CXX) $^ -o $(TARGET)

$(BUILDDIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INC) -c -o $@ $<

$(BUILDDIR)/%.o: $(SRCDIR)/%.cu
	@mkdir -p $(BUILDDIR)
	$(NVCC) $(NVCCFLAGS) $(INC) -c -o $@ $<

clean:
	$(RM) -rf $(BUILDDIR) $(LIBDIR) $(TARGET)

.PHONY: clean
