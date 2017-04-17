NVCC=nvcc
NVCCFLAGS= -std=c++11 --use_fast_math -arch=sm_35

INC=-I$(HOME)/software/nccl/build/include
LIB=-L$(HOME)/software/nccl/build/lib

all: bandwidthTest

bandwidthTest: bandwidthTest.cu
	$(NVCC) $(NVCCFLAGS) $(INC) $(LIB) $^ -o $@

clean:
	rm -f bandwidthTest
	rm -f *.o
