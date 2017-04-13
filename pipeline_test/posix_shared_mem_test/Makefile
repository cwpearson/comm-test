# the shared memory example from OSC 9e Sec. 3.5.1, Fig. 3.17, 3.18

# this must use GNU Make ("make" on Linux and Mac OS X, "gmake" on Solaris)

# compiler options -- C99 with warnings
OPT_GCC =

# compiler options and libraries for Linux, Mac OS X or Solaris

OPT = -D_XOPEN_SOURCE=700
LIB = -lrt -g -G

INC=-I$(HOME)/software/nccl/build/includeN
NLIB=-L$(HOME)/software/nccl/build/lib


all: cons prod

cons: shm-posix-consumer.c
	nvcc $(OPT_GCC) $(OPT) $(INC) $(NLIB) -o cons shm-posix-consumer.cu  $(LIB)

prod: shm-posix-producer.c
	nvcc $(OPT_GCC) $(OPT) $(INC) $(NLIB) -o prod shm-posix-producer.cu  $(LIB)

run: cons prod
	./prod
	./cons

clean:
	rm -f cons prod
