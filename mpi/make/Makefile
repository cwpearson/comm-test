CC=mpicc
CFLAGS=-Wall -Wextra -Wshadow -pedantic

all: mpi-test

mpi-test: mpi_test.o
	$(CC) $^ -o $@

mpi_test.o: mpi_test.c
	$(CC) $(CFLAGS) $^ -c -o $@

clean:
	rm -f mpi-test
	rm -f *.o
