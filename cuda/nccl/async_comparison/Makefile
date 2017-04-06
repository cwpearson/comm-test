NVCC=nvcc
NVCCFLAGS= -std=c++11 --use_fast_math -arch=sm_35

INC=-I$(HOME)/software/nccl/build/include
LIB=-L$(HOME)/software/nccl/build/lib

all: cuda_metrics_test

cuda_metrics_test: dataOverlapTest.cu
	$(NVCC) $(NVCCFLAGS) $(INC) $(LIB) $^ -o $@

clean:
	rm -f cuda_metrics_test
	rm -f *.o