NVCC=nvcc
NVCCFLAGS= -std=c++11 --use_fast_math -arch=sm_35

INC=-I$(HOME)/software/nccl/build/include
LIB=-L$(HOME)/software/nccl/build/lib

all: nccl-test

nccl-test: nccl_test.cu
	$(NVCC) $(NVCCFLAGS) $(INC) $(LIB) $^ -o $@ -lnccl

clean:
	rm -f nccl-test
	rm -f *.o
