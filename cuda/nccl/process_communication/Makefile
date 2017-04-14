NVCC=nvcc
NVCCFLAGS= -std=c++11 --use_fast_math -arch=sm_35

INC=-I$(HOME)/software/nccl/build/include
LIB=-L$(HOME)/software/nccl/build/lib

all: p1 p2 p3

p1: p1.cu
	$(NVCC) $(NVCCFLAGS) $(INC) $(LIB) $^ -o $@

p2: p2.cu
	$(NVCC) $(NVCCFLAGS) $(INC) $(LIB) $^ -o $@

p3: p3.cu
	$(NVCC) $(NVCCFLAGS) $(INC) $(LIB) $^ -o $@


clean:
	rm -f p1
	rm -f p2
	rm -f p3
	rm -f *.o
