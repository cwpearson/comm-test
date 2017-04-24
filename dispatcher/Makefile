# the shared memory example from OSC 9e Sec. 3.5.1, Fig. 3.17, 3.18

# this must use GNU Make ("make" on Linux and Mac OS X, "gmake" on Solaris)

# compiler options -- C99 with warnings
OPT_GCC = -g 

# compiler options and libraries for Linux, Mac OS X or Solaris

OPT = -D_XOPEN_SOURCE=700
LIB = -lrt -pthread

all: cons prod

mainTest: dispatcher
	gcc $(OPT_GCC) hashTable.o dispatcher.o main.c -o trial $(LIB)

dispatcher: hashTable.o
	gcc $(OPT_GCC) hashTable.o -c dispatcher.c $(LIB)

hashTable.o : 
	gcc $(OPT_GCC) -c hashTable.c $(LIB)

cons: shm-posix-consumer.cu
	nvcc $(OPT_GCC) $(OPT) -o cons shm-posix-consumer.cu  $(LIB)

prod: shm-posix-producer.cu
	nvcc $(OPT_GCC) $(OPT) -o prod shm-posix-producer.cu  $(LIB)

run: cons proddos2
	./prod
	./cons

clean:
	rm -f cons prod trial
	rm -f *.o
